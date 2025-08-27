# google genai async client
from google import genai
from google.genai.types import Blob, LiveConnectConfig, Modality, SpeechConfig, VoiceConfig, PrebuiltVoiceConfig
import asyncio
import base64
import json
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import base64
import numpy as np
import asyncio
import json
from pydub import AudioSegment
from scipy.signal import resample_poly



app = FastAPI()
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

@app.get("/")
async def index():
    return FileResponse("static/index.html")

# --- Gemini Function Calling Tool for Vector Search ---
from google.genai.types import Tool, FunctionDeclaration, FunctionResponse

vector_search_tool = Tool(
    function_declarations=[
        FunctionDeclaration(
            name="vector_search",
            description="Searches MongoDB Atlas vector collection for content relevant to the user's query.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The user's query to search for."},
                    "top_k": {"type": "integer", "description": "Number of top matches to return.", "default": 3}
                },
                "required": ["query"]
            }
        )
    ]
)
# --- MongoDB Atlas Vector Search Integration ---
from pymongo import MongoClient
import os


# Set these as needed (updated for your collection)
MONGO_URI = os.environ.get("MONGO_ATLAS_URI", "")
MONGO_DB = os.environ.get("MONGODB_NAME", "hownow_assistant_development")
MONGO_COLLECTION = os.environ.get("MONGO_COLLECTION_NAME", "file_chunks")

# Connect to MongoDB Atlas
mongo_client = MongoClient(MONGO_URI)
vector_collection = mongo_client[MONGO_DB][MONGO_COLLECTION]

# Example: Use Gemini embedding API (or any embedding model you have)
import openai
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
openai.api_key = OPENAI_API_KEY

def get_text_embedding(text):
    """
    Use OpenAI's text-embedding-3-large model to get the embedding for the input text.
    Returns a list of floats (embedding vector).
    """
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    # The API returns a list of data objects, each with an 'embedding' field
    embedding = response.data[0].embedding
    return embedding

def find_best_match(query_text, top_k=100):
    query_embedding = get_text_embedding(query_text)
    print(f"Query embedding (len {len(query_embedding)}): {query_embedding[:5]}...")
    # MongoDB Atlas vector search (requires Atlas Search index on 'embedding' field)
    pipeline = [
        {
            "$vectorSearch": {
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 100,
                "limit": top_k,
                "index": "vector_index"  # Replace with your Atlas Search index name
            }
        }
    ]
    results = list(vector_collection.aggregate(pipeline))
    print("Vector search results:")
    for result in results:
        print(f" - {result.get('text', '')} (score: {result.get('score', 0)})")
    print(f"Found {len(results)} results")
    return results

# Example FastAPI endpoint for testing
@app.post("/vector_search")
async def vector_search_api(payload: dict):
    query = payload.get("query", "")
    matches = find_best_match(query)
    return {"matches": matches}
# server.py


# --- CONFIG: change if required ---
# MODEL: choose a Live-compatible model you have access to
LIVE_MODEL = os.environ.get("GEMINI_LIVE_MODEL", "models/gemini-live-2.5-flash-preview")
# API key usage: set GEMINI_API_KEY in server env
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY environment variable with your Gemini Developer API key.")

# Initialize genai client for async use. We set api_key and http_options api_version v1beta for Live.
client = genai.Client(api_key=API_KEY)

# def to_pcm16_base64(raw_bytes, input_rate=24000, target_rate=16000):
#     """
#     Convert raw PCM16 bytes to 16kHz PCM16 LE mono and return base64 string.
#     """
#     audio = AudioSegment(
#         data=raw_bytes,
#         sample_width=2,       # PCM16 = 2 bytes
#         frame_rate=input_rate,
#         channels=1
#     )
#     audio = audio.set_frame_rate(target_rate).set_channels(1)
#     return base64.b64encode(audio.raw_data).decode("ascii")
def to_pcm16_base64(raw_bytes, input_rate=24000, target_rate=16000):
    # Interpret bytes as int16
    audio_np = np.frombuffer(raw_bytes, dtype=np.int16)

    # Resample using polyphase filtering
    gcd = np.gcd(input_rate, target_rate)
    up = target_rate // gcd
    down = input_rate // gcd
    resampled = resample_poly(audio_np, up, down)

    # Convert back to int16
    resampled = np.clip(resampled, -32768, 32767).astype(np.int16)

    # Encode base64
    return base64.b64encode(resampled.tobytes()).decode("ascii")

# Helper: convert base64 audio payload to bytes
def b64_to_bytes(b64str: str) -> bytes:
    return base64.b64decode(b64str.split(",")[-1] if "," in b64str else b64str)

from starlette.websockets import WebSocket, WebSocketDisconnect

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    # Create the Live session
    # We ask Gemini to produce AUDIO responses as well (so it returns generated speech)
    from google.genai.types import ContextWindowCompressionConfig, SlidingWindow

    live_config = LiveConnectConfig(
        response_modalities=[Modality.AUDIO],
        speech_config=SpeechConfig(
            voice_config=VoiceConfig(
                prebuilt_voice_config=PrebuiltVoiceConfig(voice_name="Aoede")
            )
        ),
        system_instruction={
            "role": "system",
            "parts": [
                {"text": "You are a professional assistant, here to help the user. For any user question that may require external knowledge or content lookup, always call the 'vector_search' function with the user's query before answering."}
            ]
        },
        context_window_compression=ContextWindowCompressionConfig(
            sliding_window=SlidingWindow()
        ),
        tools=[vector_search_tool]
    )

    print("Starting Live session with model", LIVE_MODEL)
    # Use async context to create a live session that we can send/receive to
    async with client.aio.live.connect(model=LIVE_MODEL, config=live_config) as session:
        # two tasks: forwarding from browser -> gemini, and gemini -> browser
        async def from_browser_to_gemini():
            active_turn = False
            turn_count = 0
            try:
                while True:
                    msg = await ws.receive()
                    if msg["type"] == "websocket.receive":
                        if "text" in msg:
                            payload = json.loads(msg["text"])
                            print(f"[SERVER] Received from browser: {payload.get('type')} | Payload: {payload}")
                            if payload.get("type") == "activityStart":
                                active_turn = True
                                turn_count += 1
                                print(f"[SERVER] Turn {turn_count} started")
                            elif payload.get("type") == "activityEnd":
                                if active_turn:
                                    await session.send_realtime_input(audio_stream_end=True)
                                    print(f"[SERVER] Turn {turn_count} ended, sent audio_stream_end=True")
                                    # Send a clientContent message with prompt (if provided) or empty text
                                    prompt = payload.get("prompt", "...")
                                    print(f"[SERVER] Sending clientContent after activityEnd, prompt: '{prompt}'")
                                    await session.send_client_content(turns=[{"role": "user", "parts": [{"text": prompt}]}], turn_complete=True)
                                active_turn = False
                            elif payload.get("type") == "text":
                                print(f"[SERVER] Received text: {payload['data']}")
                                await session.send_client_content(turns=[{"role":"user","parts":[{"text":payload["data"]}]}], turn_complete=True)
                        elif "bytes" in msg:
                            if active_turn:
                                # audio_bytes = msg["bytes"]
                                # print(f"[SERVER] Forwarding audio chunk to Gemini (turn {turn_count}), size: {len(audio_bytes)} bytes")
                                # await session.send_realtime_input(audio=Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000"))
                                audio_bytes = msg["bytes"]
                                # Resample to 16kHz PCM16 mono if needed
                                # Assume browser sends PCM16 at 48kHz
                                audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
                                resampled = resample_poly(audio_np, 1, 3)  # 48kHz -> 16kHz
                                resampled = np.clip(resampled, -32768, 32767).astype(np.int16)
                                audio_bytes_16k = resampled.tobytes()
                                print(f"[SERVER] Forwarding audio chunk to Gemini (turn {turn_count}), size: {len(audio_bytes_16k)} bytes")
                                await session.send_realtime_input(audio=Blob(data=audio_bytes_16k, mime_type="audio/pcm;rate=16000"))                                
                            else:
                                print("[SERVER] Ignored audio chunk (not in active turn)")
                    elif msg["type"] == "websocket.disconnect":
                        print("[SERVER] WebSocket disconnected from browser")
                        break
            except WebSocketDisconnect:
                print("[SERVER] WebSocketDisconnect caught in from_browser_to_gemini")
                return

        # async def from_gemini_to_browser():
        #     try:
            #     async for message in session.receive():
            #         # print("Received from Gemini:", message)
            #         # # message is high-level object; inspect fields for 'audio' or 'text'
            #         # # The Python SDK returns attributes like message.audio or message.text (see docs)
            #         # # We'll convert audio bytes to base64 and send as JSON: {"type":"audio","data":"<b64>"}
            #         # # and text as {"type":"text","data":"..."}
            #         # # NOTE: check actual SDK message fields in your installed version; adjust attribute names if needed.
            #         # if getattr(message, "audio", None):
            #         #     # message.audio is usually types.Blob with .data bytes
            #         #     audio_bytes = message.audio.data
            #         #     # b64 = base64.b64encode(audio_bytes).decode("ascii")
            #         #     # await ws.send_text(json.dumps({"type":"audio","data": b64}))
            #         #     print(f"Received {len(audio_bytes)} bytes audio from Gemini")
            #         #     b64_audio = to_pcm16_base64(audio_bytes, input_rate=24000, target_rate=16000)
            #         #     await ws.send_json({
            #         #         "parts": [
            #         #             {"type": "audio", "data": b64_audio}
            #         #         ]
            #         #     })
            #         # if getattr(message, "text", None):
            #         #     await ws.send_text(json.dumps({"type":"text","data": message.text}))
            #         if message.model_turn:
            #             print("model_turn:", message.model_turn)
            #             # model_turn has parts which may include text, audio, etc.
            #             parts = []
            #             for part in message.model_turn.parts:
            #                 print(" part:", part)
            #                 # Handle text responses
            #                 if getattr(part, "text", None):
            #                     parts.append({"type": "text", "text": part.text})

            #                 # Handle audio responses
            #                 elif getattr(part, "inline_data", None) and \
            #                     part.inline_data.mime_type.startswith("audio/pcm"):
            #                     raw_bytes = part.inline_data.data
            #                     # Convert 24k PCM16 → 16k PCM16 before sending
            #                     b64_audio = to_pcm16_base64(raw_bytes, input_rate=24000, target_rate=16000)
            #                     parts.append({"type": "audio", "data": b64_audio})

            #         if parts:
            #             await ws.send_json({"parts": parts})
            #             # You may also receive transcription messages, interrupted, turnComplete flags etc.
            # except Exception as e:
            #     # session.receive may raise when session ended
            #     await ws.close()
            #     return
        async def from_gemini_to_browser():
            response_turn = 0
            try:
                while True:
                    async for message in session.receive():
                        print(f"[SERVER] Received from Gemini: {message}")

                        # Handle Gemini function/tool calls
                        if getattr(message, "tool_call", None):
                            tool_call = message.tool_call
                            if hasattr(tool_call, "function_calls"):
                                for call in tool_call.function_calls:
                                    if call.name == "vector_search":
                                        query = call.args.get("query", "")
                                        top_k = call.args.get("top_k", 100)
                                        print(f"[SERVER] Gemini requested vector_search: query='{query}', top_k={top_k}")
                                        matches = find_best_match(query, top_k=top_k)
                                        for m in matches:
                                            print("Mongo embedding_text:", m.get("text"))
                                        # Prepare response as a string summary (or pass full docs as needed)
                                        # summary = "\n".join([str(m.get("content", m)) for m in matches])
                                        summary = "\n".join([str(m.get("text", "")) for m in matches])
                                        print("MongoDB search results:", summary)
                                        # await session.send_tool_response(
                                        #     FunctionResponse(
                                        #         name="vector_search",
                                        #         response={"results": summary},
                                        #         id=call.id
                                        #     )
                                        # )
                                        print('_'*80)
                                        function_response=FunctionResponse(
                                                name='vector_search',
                                                response={'result': summary},
                                                id=call.id,
                                            )
                                        # print("function_response", function_response)
                                        await session.send_tool_response(
                                            function_responses=function_response
                                        )

                                        print('_'*80)                                        
                                        print(f"[SERVER] Sent vector_search results to Gemini.")
                            continue

                        # Case 1: top-level model_turn
                        if getattr(message, "model_turn", None):
                            response_turn += 1
                            print(f"[SERVER] Gemini model_turn {response_turn} parts: {message.model_turn.parts}")
                            for part in message.model_turn.parts:
                                print(f"[SERVER] Gemini part: {part}")
                                if getattr(part, "inline_data", None) and part.inline_data.mime_type.startswith("audio/pcm"):
                                    raw_bytes = part.inline_data.data
                                    print(f"[SERVER] Sending audio to browser, size: {len(raw_bytes)} bytes (turn {response_turn})")
                                    await ws.send_bytes(raw_bytes)
                                elif getattr(part, "text", None):
                                    print(f"[SERVER] Sending text to browser: {part.text}")
                                    await ws.send_json({"type": "text", "text": part.text})

                        # Case 2: server_content contains a model_turn
                        elif getattr(message, "server_content", None) and getattr(message.server_content, "model_turn", None):
                            response_turn += 1
                            print(f"[SERVER] Gemini server_content.model_turn {response_turn} parts: {message.server_content.model_turn.parts}")
                            for part in message.server_content.model_turn.parts:
                                print(f"[SERVER] Gemini part: {part}")
                                if getattr(part, "inline_data", None) and part.inline_data.mime_type.startswith("audio/pcm"):
                                    raw_bytes = part.inline_data.data
                                    print(f"[SERVER] Sending audio to browser, size: {len(raw_bytes)} bytes (turn {response_turn})")
                                    await ws.send_bytes(raw_bytes)
                                elif getattr(part, "text", None):
                                    print(f"[SERVER] Sending text to browser: {part.text}")
                                    await ws.send_json({"type": "text", "text": part.text})
                        

                        sc = getattr(message, "server_content", None)
                        if sc and getattr(sc, "turn_complete", False):
                            print(f"[SERVER] Gemini finished speaking for turn {response_turn}, waiting for next input…")
                            # DO NOT close session

            except Exception as e:
                print(f"[SERVER] Error in from_gemini_to_browser: {e}")
                return


        # Run both tasks concurrently until one finishes
        # tasks = [
        #     asyncio.create_task(from_browser_to_gemini()),
        #     asyncio.create_task(from_gemini_to_browser())
        # ]
        # done, pending = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
        # for t in pending:
        #     t.cancel()
        await asyncio.gather(
            from_browser_to_gemini(),
            from_gemini_to_browser()
        )
        

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

# from google import genai
# client = genai.Client(api_key="AIzaSyAZFid1QvnKseBW3_S5yR7PxadJfO51mwY")
# # Try a simple generate (non-live) — replace with a model you know you have access to from console
# try:
#     r = client.models.generate_content(model="models/gemini-2.0-flash-001", contents="Hello")
#     print("OK:", r.text[:200])
# except Exception as e:
#     print("generate_content failed:", e)
    

# import os, traceback, json
# from google import genai
# import asyncio

# api_key = os.environ.get("GEMINI_API_KEY")  # or set directly
# client = genai.Client(api_key=api_key)

# LIVE_MODEL = "models/gemini-2.0-flash-live-001"  # change if needed

# from google.genai.types import LiveConnectConfig, Modality, SpeechConfig, VoiceConfig, PrebuiltVoiceConfig

# live_config = LiveConnectConfig(
#     response_modalities=[Modality.AUDIO],
#     speech_config=SpeechConfig(
#         voice_config=VoiceConfig(
#             prebuilt_voice_config=PrebuiltVoiceConfig(voice_name="Aoede")
#         )
#     )
# )

# async def test_live():
#     try:
#         async with client.aio.live.connect(model=LIVE_MODEL, config=live_config) as session:
#             print("Connected OK (unexpected if no Live access).")
#     except Exception as e:
#         print("Live connect failed:", repr(e))
#         traceback.print_exc()

# asyncio.run(test_live())

