from chatbot_logic import get_chatbot_logic_response  # Replace with actual function name
import os

from pydub import AudioSegment

AudioSegment.converter = r"C:\ffmpeg\ffmpeg-7.1-essentials_build\bin\ffmpeg.exe"

import time
import logging
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, UploadFile, HTTPException
from fastapi.responses import FileResponse
import whisper  # OpenAI Whisper
import edge_tts  # Microsoft Edge TTS
import soundfile as sf  # Audio processing

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VoiceAI")

# Configuration
AUDIO_TEMP_DIR = "audio_temp"
os.makedirs(AUDIO_TEMP_DIR, exist_ok=True)

class ConversationManager:
    def __init__(self):
        self.context = {}

    async def get_response(self, text: str, user_id: str = "default_user") -> str:
        """
        Get a chatbot response for the given text, maintaining conversation context per user.
        """
        if user_id not in self.context:
            self.context[user_id] = []
        
        # Add the user input to the conversation context
        self.context[user_id].append({"role": "user", "content": text})
        
        # Create a context string if needed (e.g., by joining previous user messages)
        context_str = "\n".join([msg["content"] for msg in self.context[user_id] if msg["role"] == "user"])
        
        # Asynchronously get the chatbot response using the asynchronous function
        chatbot_response = await get_chatbot_logic_response(text, context=context_str)
        
        # Save the assistant's response in the context
        self.context[user_id].append({"role": "assistant", "content": chatbot_response})
        
        return chatbot_response

# --- Speech-to-Text (STT) Implementation ---
class SpeechToText:
    def __init__(self):
        self.whisper_model = whisper.load_model("base")  # small/medium/large for better accuracy
        
    async def transcribe(self, audio_path: str) -> str:
        """Convert speech to text using Whisper"""
        try:
            result = self.whisper_model.transcribe(audio_path)
            return result["text"]
        except Exception as e:
            logger.error(f"STT Error: {str(e)}")
            raise

# --- Text-to-Speech (TTS) Implementation ---
class TextToSpeech:
    async def synthesize(self, text: str, provider: str = "edge") -> str:
        """Convert text to speech using selected provider"""
        try:
            output_path = os.path.join(AUDIO_TEMP_DIR, f"output_{int(time.time())}.wav")
            
            if provider == "edge":
                return await self._edge_tts(text, output_path)
            elif provider == "elevenlabs":
                return await self._elevenlabs_tts(text, output_path)
            else:
                raise ValueError("Unsupported TTS provider")
                
        except Exception as e:
            logger.error(f"TTS Error: {str(e)}")
            raise

    async def _edge_tts(self, text: str, output_path: str) -> str:
        """Microsoft Edge TTS implementation"""
        communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
        await communicate.save(output_path)
        return output_path

    async def _elevenlabs_tts(self, text: str, output_path: str) -> str:
        """ElevenLabs TTS implementation (requires API key)"""
        # Implementation example using ElevenLabs SDK
        from elevenlabs import generate, save
        audio = generate(
            text=text,
            voice="Bella",
            model="eleven_monolingual_v1",
            api_key=os.getenv("ELEVEN_LABS_API_KEY")
        )
        save(audio, output_path)
        return output_path
    
conversation_manager = ConversationManager()

@app.post("/voice-command")
async def process_voice_command(file: UploadFile, user_id: str = "default_user"):
    try:
        # Save uploaded audio
        input_path = os.path.join(AUDIO_TEMP_DIR, f"input_{int(time.time())}.wav")
        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Convert speech to text
        stt = SpeechToText()
        text_command = await stt.transcribe(input_path)
        logger.info(f"Recognized command: {text_command}")

        # Get chatbot response with context
        chatbot_response = await conversation_manager.get_response(text_command, user_id)
        logger.info(f"Chatbot response: {chatbot_response}")
        
        # Convert response to speech
        tts = TextToSpeech()
        output_path = await tts.synthesize(chatbot_response)

        return FileResponse(output_path, media_type="audio/wav")

    except Exception as e:
        logger.error(f"Voice processing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Voice processing failed")

@app.websocket("/voice-stream/{user_id}")
async def voice_stream(websocket: WebSocket, user_id: str):
    """Real-time voice processing with streaming TTS"""
    await websocket.accept()
    try:
        conversation_manager = ConversationManager()
        tts = TextToSpeech()
        stt = SpeechToText()
        
        while True:
            # Receive audio chunk (max 5s duration)
            audio_data = await websocket.receive_bytes()
            
            # Save temporary audio file
            input_path = os.path.join(AUDIO_TEMP_DIR, f"stream_{user_id}_{int(time.time())}.wav")
            with open(input_path, "wb") as f:
                f.write(audio_data)

            # STT Processing
            text = await stt.transcribe(input_path)
            logger.info(f"User {user_id}: {text}")

            # Get chatbot response
            response = await conversation_manager.get_response(text, user_id)
            
             #Stream TTS response using edge_tts
            communicate = edge_tts.Communicate(response, "en-US-AriaNeural")
            try:
                async for chunk in communicate.stream():
                    if chunk.get("type") == "audio":
                        await websocket.send_bytes(chunk.get("data"))
            except Exception as tts_error:
                logger.error(f"TTS streaming error: {str(tts_error)}")
                # Optionally, send an error message to the client or break out of the loop
                await websocket.send_text("[TTS_ERROR]")
                break
                    
            # Send end-of-stream marker
            await websocket.send_text("[EOS]")

    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.close()

        