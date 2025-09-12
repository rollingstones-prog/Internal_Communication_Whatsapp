# ElevenLabs API integration for TTS and STT
# Add your ElevenLabs API key to .env as ELEVENLABS_API_KEY
import requests
import os
from dotenv import load_dotenv


load_dotenv(override=True)
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"

# Default voice_id (can be customized)
ELEVENLABS_VOICE_ID = os.getenv("ELEVEN_VOICE_ID")


def tts_to_mp3(text, out_path, voice_id=ELEVENLABS_VOICE_ID):
    """
    Convert text to speech using ElevenLabs API and save as MP3.
    """
    url = ELEVENLABS_TTS_URL.format(voice_id=voice_id)
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.5},
        "output_format": "mp3"
    }
    print(ELEVENLABS_API_KEY)
    print(voice_id)
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(response.content)
            return out_path
        else:
            print("ElevenLabs TTS error:", response.text)
            return None
    except Exception as e:
        print("ElevenLabs TTS exception:", str(e))
        return None


def stt_from_mp3(audio_path):
    """
    Transcribe audio using ElevenLabs API (MP3, OGG, WAV, etc).
    """
    url = ELEVENLABS_STT_URL
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY
    }
    try:
        with open(audio_path, "rb") as f:
            files = {"audio": f}
            response = requests.post(url, headers=headers, files=files, timeout=60)
        if response.status_code == 200:
            result = response.json()
            return result.get("text", "")
        else:
            print("ElevenLabs STT error:", response.text)
            return "Audio transcription me temporary issue hai."
    except Exception as e:
        print("ElevenLabs STT exception:", str(e))
        return "Audio transcription me temporary issue hai."
