import subprocess
import tempfile
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from datetime import datetime
import json
import os
import requests
from dotenv import load_dotenv

# 🛑 CRITICAL FIX: db.py se zaroori async functions aur models import kiye
from db import get_new_messages, get_session, Employee 


# .env file load karein (Agar local development mein hain)
load_dotenv(override=True)

# -------------------
# Init FastAPI & CORS
# -------------------
app = FastAPI() # <--- Yeh hi woh 'app' object hai jo uvicorn dhundhta hai.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------
# DB & File Setup
# -------------------
# DB_URL ko yahan se hata diya gaya hai.
EMPLOYEES_FILE = Path("./employees.json")
REPORTS_DIR = Path("./reports")
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# ElevenLabs Setup (Wahi rakha gaya hai)
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"


# -------------------
# Database Functions (db.py ke async functions use honge)
# -------------------

async def read_db_events():
    """db.py se WhatsAppInbox ke messages load karta hai aur unhe 'event' format mein badalta hai."""
    try:
        # get_new_messages() ab sirf unprocessed messages laata hai
        inbox_messages = await get_new_messages() 
        
        events = []
        for msg in inbox_messages:
            # WhatsAppInbox data ko dashboard ke expected event format (WA_RECV) mein badlein
            
            # NOTE: Agar db.py mein 'at' key nahi hai, toh yeh line masla kar sakti hai.
            # Lekin aapke db.py mein 'created_at' column hai, toh hum assume karte hain ki woh aa raha hai.
            
            # Agar 'at' key available nahi hai, toh default timestamp use karein
            ts = msg.get('at', datetime.utcnow())
            ts_iso = ts.isoformat() + "Z" if isinstance(ts, datetime) else datetime.utcnow().isoformat() + "Z"
            
            events.append({
                "kind": "WA_RECV",
                "employee": find_employee_name_by_msisdn(msg['phone']),
                "msisdn": msg['phone'],
                "at": ts_iso, 
                "payload": {
                    "type": "text",
                    "text": {"body": msg['message_text']}
                }
            })
        
        # Data ko chronological order (shuru se aakhir tak) mein rakhein.
        return sorted(events, key=lambda x: x.get('at', ""))
        
    except Exception as e:
        print(f"❌ Database Read Error using db.py: {e}")
        return []


# -------------------
# Helper Functions (Employee data local file se)
# -------------------

def load_employees():
    """Load employees.json and return as dict."""
    if not EMPLOYEES_FILE.exists():
        return {}
    with open(EMPLOYEES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def find_employee_name_by_msisdn(msisdn: str):
    employees = load_employees()
    for name, info in employees.items():
        if isinstance(info, dict) and info.get("msisdn") == msisdn:
            return name
        elif info == msisdn: # Fallback for old employee.json structure
            return name
    return msisdn

# -------------------
# ElevenLabs API Functions (Wahi rakha gaya hai)
# -------------------

def tts_to_mp3(text, out_path, voice_id=ELEVENLABS_VOICE_ID):
    url = ELEVENLABS_TTS_URL.format(voice_id=voice_id)
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
    payload = {"text": text, "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}, "output_format": "mp3"}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(resp.content)
            return str(out_path)
        print(f"TTS API Error: {resp.status_code}, {resp.text}")
        return None
    except Exception as e:
        print("TTS error:", e)
        return None

def stt_from_audio(audio_path: Path):
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    try:
        audio_path = Path(audio_path)
        use_path = audio_path
        if audio_path.suffix.lower() in [".ogg", ".opus", ".wav"]:
            out_path = Path(tempfile.gettempdir()) / (audio_path.stem + ".mp3")
            if not audio_path.is_file():
                raise ValueError(f"Invalid audio file: {audio_path.name}")
            cmd = ["ffmpeg", "-y", "-i", str(audio_path.resolve()), str(out_path.resolve())]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            use_path = out_path

        with open(use_path, "rb") as f:
            files = {"file": f}
            data = {"model_id": "scribe_v1"}
            resp = requests.post(ELEVENLABS_STT_URL, headers=headers, data=data, files=files, timeout=60)

        if resp.status_code == 200:
            result = resp.json()
            return result.get("text", "").strip()
        else:
            print("❌ ElevenLabs STT error:", resp.status_code, resp.text)
            return "Audio transcription me temporary issue hai."

    except Exception as e:
        print("❌ ElevenLabs STT exception:", str(e))
        return "Audio transcription me temporary issue hai."


# -------------------
# API Endpoints 
# -------------------

@app.get("/chats/all")
async def get_all_chats():
    # Ab DB se events load honge
    events = await read_db_events() 
    employees = {}
    
    for e in events:
        kind = e.get("kind")
        if kind != "WA_RECV": 
            continue

        emp_name = e.get("employee")
        msisdn = e.get("msisdn")
        if not emp_name:
            continue

        payload = e.get("payload", {})
        last_msg = payload.get("text", {}).get("body") or "📎 Media"
        last_ts = e.get("at")
        
        # Hamesha latest message se update karein
        employees[emp_name] = {
            "name": emp_name,
            "msisdn": msisdn,
            "lastMessage": last_msg,
            "lastTimestamp": last_ts,
            "avatar": f"https://api.dicebear.com/7.x/identicon/svg?seed={emp_name}",
            "online": True
        }

    return {"employees": list(employees.values())}


@app.get("/chats/{employee}")
async def get_employee_chat(employee: str):
    # Ab DB se events load honge
    events = await read_db_events() 
    chat = []
    
    for e in events:
        kind = e.get("kind")
        emp_name = (e.get("employee") or e.get("msisdn") or "").lower()
        if emp_name != employee.lower() or kind != "WA_RECV":
            continue

        payload = e.get("payload", {})
        
        chat.append({
            "sender": "Employee",
            "type": payload.get("type", "text"),
            "text": payload.get("text", {}).get("body"),
            "filename": None, 
            "timestamp": e.get("at")
        })

    # Sort by time
    chat = sorted(chat, key=lambda x: x.get("timestamp") or "")

    return {
        "employee": employee,
        "messages": chat
    }


@app.post("/send")
async def send_message(request: Request):
    data = await request.json()
    employee = data.get("employee")
    message = data.get("message", {})
    if not employee or not message:
        return {"error": "employee and message are required"}
    
    # TODO: WhatsApp API call to send message & Database mein WA_SEND event save karein
    print(f"DEBUG: Message sent to {employee} and should be logged to DB.")

    return {"status": "ok", "message": "Message sent/logged (DB implementation needed)"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    safe_filename = Path(file.filename).name
    file_path = UPLOAD_DIR / safe_filename
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # TODO: Database mein UPLOAD event save karein
    print(f"DEBUG: File uploaded: {safe_filename} and should be logged to DB.")

    return {"status": "ok", "file": safe_filename, "path": str(file_path)}


@app.get("/report/{employee}")
def get_report(employee: str):
    report_file = REPORTS_DIR / f"{employee}.txt"
    if not report_file.exists():
        return {"employee": employee, "report": "No report found"}
    return {"employee": employee, "report": report_file.read_text(encoding="utf-8")}

@app.post("/tts")
async def text_to_speech(request: Request):
    data = await request.json()
    text = data.get("text")
    if not text:
        return {"error": "No text provided"}
    out_path = UPLOAD_DIR / f"tts_{int(datetime.utcnow().timestamp())}.mp3"
    file_path = tts_to_mp3(text, out_path)
    return {"file": file_path}


@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    safe_filename = Path(file.filename).name
    audio_path = UPLOAD_DIR / safe_filename
    with open(audio_path, "wb") as f:
        f.write(await file.read())
    text = stt_from_audio(audio_path)
    return {"text": text}
