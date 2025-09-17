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
from supabase import create_client
import os
from utils import now_iso


from supabase import create_client, Client
from dotenv import load_dotenv
load_dotenv(override=True)
# .env file me ye keys honi chahiye:
# SUPABASE_URL=https://xxxx.supabase.co
# SUPABASE_KEY=eyJhbGciOi...

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# -------------------
# Init FastAPI
# -------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
EMPLOYEES_FILE = Path("./employees.json")

def load_employees():
    """
    Load employees.json and return as dict.
    """
    if not EMPLOYEES_FILE.exists():
        return {}
    with open(EMPLOYEES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def find_employee_name_by_msisdn(msisdn: str):
    """
    Given a WhatsApp msisdn, return the mapped employee name from employees.json.
    If not found, fallback to msisdn.
    """
    employees = load_employees()
    for name, info in employees.items():
        if info.get("msisdn") == msisdn:
            return name
    return msisdn
def read_events(limit=200):
    try:
        # Try with "at" if it exists
        resp = supabase.table("events").select("*").order("at", desc=True).limit(limit).execute()
        if resp.data:
            return resp.data
        # fallback if at is null/empty
        resp = supabase.table("events").select("*").limit(limit).execute()
        return resp.data or []
    except Exception as e:
        print(f"❌ Supabase read error: {e}")
        return []

# -------------------
# Files & Directories
# -------------------
EVENTS_FILE = Path("./events.jsonl")
REPORTS_DIR = Path("./reports")
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# -------------------
# ElevenLabs API Setup
# -------------------
load_dotenv(override=True)
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")

ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
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
        return None
    except Exception as e:
        print("TTS error:", e)
        return None


def stt_from_audio(audio_path: Path):
    """
    Transcribe audio using ElevenLabs STT.
    Supports MP3/OGG/OPUS/WAV (auto convert with ffmpeg if needed).
    """
    ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

    headers = {"xi-api-key": ELEVENLABS_API_KEY}

    try:
        audio_path = Path(audio_path)

        # Agar file ogg/opus/wav hai to ffmpeg se mp3 bana lo
        use_path = audio_path
        if audio_path.suffix.lower() in [".ogg", ".opus", ".wav"]:
            out_path = Path(tempfile.gettempdir()) / (audio_path.stem + ".mp3")
            cmd = ["ffmpeg", "-y", "-i", str(audio_path), str(out_path)]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            use_path = out_path

        with open(use_path, "rb") as f:
            files = {"file": f}                # 👈 correct field
            data = {"model_id": "scribe_v1"}   # 👈 valid model
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

@app.get("/chats/all")
def get_all_chats():
    events = read_events()
    employees = {}

    for e in events:
        kind = e.get("kind")
        if kind not in ["WA_SEND", "WA_RECV"]:
            continue   # ✅ sirf Agent/Employee msgs lo

        emp_name = e.get("employee") or e.get("to")
        msisdn = e.get("msisdn")
        if not emp_name:
            continue

        last_msg, last_ts = None, None

        # ---- Agent → Employee ----
        if kind == "WA_SEND":
            last_msg = e.get("payload", {}).get("text", {}).get("body") or "📎 Media"
            last_ts = e.get("at")

            # ✅ sirf tab save karo jab employee ne abhi tak reply nahi kiya
            if emp_name not in employees:
                employees[emp_name] = {
                    "name": emp_name,
                    "msisdn": msisdn,
                    "lastMessage": last_msg or "No message",
                    "lastTimestamp": last_ts,
                    "avatar": f"https://api.dicebear.com/7.x/identicon/svg?seed={emp_name}",
                    "online": True
                }

        # ---- Employee → Agent ----
        elif kind == "WA_RECV":
            payload = e.get("payload", {})

            # Case 1: Direct text payload
            if isinstance(payload, dict) and "text" in payload:
                last_msg = payload.get("text", {}).get("body")
                last_ts = e.get("at")

            # Case 2: Direct messages list
            elif "messages" in payload:
                m = payload["messages"][-1]
                msg_type = m.get("type")
                last_msg = m.get("text", {}).get("body") if msg_type == "text" else "📎 Media"
                last_ts = m.get("timestamp")

            # Case 3: Webhook nested entry → changes → value → messages
            elif "entry" in payload:
                for entry in payload.get("entry", []):
                    for change in entry.get("changes", []):
                        for m in change.get("value", {}).get("messages", []):
                            msg_type = m.get("type")
                            last_msg = m.get("text", {}).get("body") if msg_type == "text" else "📎 Media"
                            last_ts = m.get("timestamp")

            # ✅ Always overwrite with employee ka msg (latest priority)
            employees[emp_name] = {
                "name": emp_name,
                "msisdn": msisdn,
                "lastMessage": last_msg or "No message",
                "lastTimestamp": last_ts,
                "avatar": f"https://api.dicebear.com/7.x/identicon/svg?seed={emp_name}",
                "online": True
            }

    return {"employees": list(employees.values())}



def find_employee_name_by_msisdn(msisdn: str):
    employees = load_employees()
    for name, info in employees.items():
        if info.get("msisdn") == msisdn:
            return name
    return msisdn


@app.get("/chats/{employee}")
def get_employee_chat(employee: str):
    events = read_events()
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

            # Case 1: Direct text
            if isinstance(payload, dict) and "text" in payload:
                chat.append({
                    "sender": "Employee",
                    "type": "text",
                    "text": payload.get("text", {}).get("body"),
                    "filename": None,
                    "timestamp": e.get("at")
                })

            # Case 2: Direct messages list
            if "messages" in payload:
                for m in payload["messages"]:
                    msg_type = m.get("type")
                    chat.append({
                        "sender": "Employee",
                        "type": msg_type,
                        "text": m.get("text", {}).get("body") if msg_type == "text" else None,
                        "filename": m.get(msg_type, {}).get("filename") if msg_type in ["document","image","audio"] else None,
                        "timestamp": m.get("timestamp")
                    })

            # Case 3: Webhook nested entry → changes → value → messages
            if "entry" in payload:
                for entry in payload.get("entry", []):
                    for change in entry.get("changes", []):
                        for m in change.get("value", {}).get("messages", []):
                            msg_type = m.get("type")
                            chat.append({
                                "sender": "Employee",
                                "type": msg_type,
                                "text": m.get("text", {}).get("body") if msg_type == "text" else None,
                                "filename": m.get(msg_type, {}).get("filename") if msg_type in ["document","image","audio"] else None,
                                "timestamp": m.get("timestamp")
                            })

    # ✅ sort by time
    chat = sorted(chat, key=lambda x: x.get("timestamp") or "")

    return {
        "employee": employee,
        "messages": chat
    }

def append_event(record: dict):
    print("👉 append_event called with:", record)   # Debug
    try:
        resp = supabase.table("events").insert(record).execute()
        print("👉 Supabase response:", resp)
    except Exception as e:
        print(f"❌ Supabase insert error: {e}")


@app.post("/send")
async def send_message(request: Request):
    data = await request.json()
    employee = data.get("employee")
    message = data.get("message", {})
    if not employee or not message:
        return {"error": "employee and message are required"}
    record = {
        "kind": "WA_SEND",
        "to": employee,
        "at": datetime.utcnow().isoformat(),
        "payload": {"text": {"body": message.get("text")}, "type": message.get("type", "text")}
    }
    supabase.table("events").insert(record).execute()

    return {"status": "ok", "saved": record}


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
    audio_path = UPLOAD_DIR / file.filename
    with open(audio_path, "wb") as f:
        f.write(await file.read())
    text = stt_from_audio(audio_path)
    return {"text": text}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload any file (PDF, DOC, XLS, Image, etc.)
    Saves into ./uploads and logs into events.jsonl
    """
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        f.write(await file.read())

    record = {
        "kind": "UPLOAD",
        "filename": file.filename,
        "path": str(file_path),
        "at": datetime.utcnow().isoformat(),
    }
    supabase.table("events").insert(record).execute()

    return {"status": "ok", "file": file.filename, "path": str(file_path)}

