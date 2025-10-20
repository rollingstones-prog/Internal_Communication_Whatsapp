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
import psycopg2 # <-- PostgreSQL library
from urllib.parse import urlparse # <-- DB URL parsing ke liye

# .env file load karein (Agar local development mein hain)
load_dotenv(override=True)

# -------------------
# Init FastAPI & CORS
# -------------------
app = FastAPI()

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
# Zaroori Environment Variables
DB_URL = os.getenv("DATABASE_URL")
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
# PostgreSQL Functions
# -------------------

def get_db_connection():
    """PostgreSQL se connection establish karta hai."""
    if not DB_URL:
        # NOTE: Agar aapka bot data save kar raha hai, toh DB_URL hona zaroori hai.
        print("❌ DATABASE_URL environment variable is not set. Cannot connect to PostgreSQL.")
        return None
    try:
        # DB URL ko parse karke psycopg2 se connect karte hain.
        url = urlparse(DB_URL)
        conn = psycopg2.connect(
            database=url.path[1:],
            user=url.username,
            password=url.password,
            host=url.hostname,
            port=url.port,
            sslmode='require' if url.scheme in ['postgres', 'postgresql'] else None
        )
        return conn
    except Exception as e:
        print(f"❌ PostgreSQL Connection Error: {e}")
        return None

def read_db_events(limit=200):
    """Database se saare events (messages) load karta hai."""
    conn = get_db_connection()
    if not conn:
        return []

    events = []
    try:
        cur = conn.cursor()
        # Hum assume kar rahe hain ki aapki table ka naam 'events' hai aur data JSON/TEXT column mein hai.
        # Aapka db.py WhatsAppInbox table dikha raha tha, lekin hum 'events' table ka standard structure use kar rahe hain
        # jo aapke bot se data save karta hoga. Agar aap WhatsAppInbox se data load kar rahe hain,
        # toh query badalni hogi. Filhaal standard 'events' table use kar rahe hain.
        cur.execute(
            # Humne yahan 'data' column assume kiya hai. Agar aapka column naam alag hai toh badal dein.
            "SELECT data FROM events ORDER BY timestamp DESC LIMIT %s",
            (limit,)
        )
        
        # Har row se JSON data load karein
        events_json = [row[0] for row in cur.fetchall()]
        # Agar DB mein data JSON string ke taur par save hua hai toh use parse karein
        events = [json.loads(e) if isinstance(e, str) else e for e in events_json]
        
        cur.close()
        conn.close()
        # Data ko hamesha chronological order (shuru se aakhir tak) mein rakhein.
        return list(reversed(events))
        
    except Exception as e:
        print(f"❌ Database Read Error: {e}")
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
        if info.get("msisdn") == msisdn:
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
# API Endpoints (read_events ko read_db_events se replace kiya gaya hai)
# -------------------

@app.get("/chats/all")
def get_all_chats():
    # Ab DB se events load honge
    events = read_db_events() 
    employees = {}

    for e in events:
        kind = e.get("kind")
        if kind not in ["WA_SEND", "WA_RECV"]:
            continue

        emp_name = e.get("employee") or e.get("to")
        msisdn = e.get("msisdn")
        if not emp_name:
            continue

        last_msg, last_ts = None, None

        # ---- Agent → Employee ----
        if kind == "WA_SEND":
            last_msg = e.get("payload", {}).get("text", {}).get("body") or "📎 Media"
            last_ts = e.get("at")
        
        # ---- Employee → Agent ----
        elif kind == "WA_RECV":
            payload = e.get("payload", {})
            # Webhook structure ko handle karne ki logic (simplified)
            if isinstance(payload, dict) and "text" in payload:
                last_msg = payload.get("text", {}).get("body")
                last_ts = e.get("at")
            elif "messages" in payload:
                m = payload["messages"][-1]
                msg_type = m.get("type")
                last_msg = m.get("text", {}).get("body") if msg_type == "text" else "📎 Media"
                last_ts = m.get("timestamp")
            elif "entry" in payload:
                for entry in payload.get("entry", []):
                    for change in entry.get("changes", []):
                        for m in change.get("value", {}).get("messages", []):
                            msg_type = m.get("type")
                            last_msg = m.get("text", {}).get("body") if msg_type == "text" else "📎 Media"
                            last_ts = m.get("timestamp")

            if not last_ts: 
                last_ts = e.get("at")

        # Hamesha latest message se update karein
        employees[emp_name] = {
            "name": emp_name,
            "msisdn": msisdn,
            "lastMessage": last_msg or "No message",
            "lastTimestamp": last_ts,
            "avatar": f"https://api.dicebear.com/7.x/identicon/svg?seed={emp_name}",
            "online": True
        }

    return {"employees": list(employees.values())}


@app.get("/chats/{employee}")
def get_employee_chat(employee: str):
    # Ab DB se events load honge
    events = read_db_events() 
    chat = []
    
    for e in events:
        kind = e.get("kind")
        emp_name = (e.get("employee") or e.get("to") or "").lower()
        if emp_name != employee.lower():
            continue

        # ---- Agent → Employee ----
        if kind == "WA_SEND":
            chat.append({
                "sender": "Agent",
                "type": e.get("payload", {}).get("type", "text"),
                "text": e.get("payload", {}).get("text", {}).get("body"),
                "filename": e.get("payload", {}).get("filename"),
                "timestamp": e.get("at")
            })

        # ---- Employee → Agent ----
        elif kind == "WA_RECV":
            payload = e.get("payload", {})
            messages = []
            
            # Simple text message
            if isinstance(payload, dict) and "text" in payload:
                 messages.append({"type": "text", "text": {"body": payload["text"]["body"]}, "timestamp": e.get("at")})
                 
            # Extract messages from different webhook structures
            if "messages" in payload:
                messages.extend(payload["messages"])
            elif "entry" in payload:
                for entry in payload.get("entry", []):
                    for change in entry.get("changes", []):
                        messages.extend(change.get("value", {}).get("messages", []))
            
            for m in messages:
                msg_type = m.get("type")
                chat.append({
                    "sender": "Employee",
                    "type": msg_type,
                    "text": m.get("text", {}).get("body") if msg_type == "text" else None,
                    "filename": m.get(msg_type, {}).get("filename") if msg_type in ["document","image","audio"] else None,
                    "timestamp": m.get("timestamp")
                })

    # Sort by time
    chat = sorted(chat, key=lambda x: x.get("timestamp") or "")

    return {
        "employee": employee,
        "messages": chat
    }

# NOTE: /send aur /upload endpoints ko ab local file ki jagah database mein event
# save karne aur WhatsApp message bhejney ki logic chahiyye. Filhaal unko 
# sirf debug ke liye rakha gaya hai, kyunki DB insert ka code yahan nahi diya jaa sakta.
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

# /report/{employee}, /tts, /stt functions are kept as is...
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
