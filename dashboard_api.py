from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from datetime import datetime
import json
import os
import requests
from dotenv import load_dotenv

from utils import append_jsonl, read_jsonl

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
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    try:
        with open(audio_path, "rb") as f:
            files = {"audio": f}
            resp = requests.post(ELEVENLABS_STT_URL, headers=headers, files=files, timeout=60)
        if resp.status_code == 200:
            return resp.json().get("text", "")
        return ""
    except Exception as e:
        print("STT error:", e)
        return ""
@app.get("/chats/all")
def get_all_chats():
    events = read_jsonl(EVENTS_FILE)
    employees = {}
    for e in events:
        entries = e.get("payload", {}).get("entry", [])
        for entry in entries:
            for change in entry.get("changes", []):
                value = change.get("value", {})
                contacts = value.get("contacts", [])
                messages = value.get("messages", [])
                for c in contacts:
                    msisdn = c.get("wa_id") or ""
                    mapped_name = find_employee_name_by_msisdn(msisdn)
                    if not mapped_name:
                        continue
                    last_msg, last_ts = None, None
                    if messages:
                        m = messages[-1]
                        last_msg = m.get("text", {}).get("body") if m.get("type") == "text" else "📎 Media"
                        last_ts = m.get("timestamp")
                    employees[mapped_name] = {
                        "name": mapped_name,
                        "msisdn": msisdn,
                        "lastMessage": last_msg or "No message",
                        "lastTimestamp": last_ts,
                        "avatar": f"https://api.dicebear.com/7.x/identicon/svg?seed={mapped_name}",
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
    events = read_jsonl(EVENTS_FILE)
    chat = []

    for e in events:
        kind = e.get("kind")

        # ✅ Agent → Employee messages
        if kind == "WA_SEND" and (e.get("employee") or e.get("to", "")).lower() == employee.lower():
            chat.append({
                "sender": "Agent",
                "text": e.get("payload", {}).get("text", {}).get("body", ""),
                "timestamp": e.get("at")
            })

        # ✅ Employee → Agent messages
        elif kind == "WA_RECV" and e.get("employee") and e["employee"].lower() == employee.lower():
            # Filter only actual messages, skip delivery receipts
            for entry in e.get("payload", {}).get("entry", []):
                for change in entry.get("changes", []):
                    value = change.get("value", {})
                    if "messages" in value:  # actual incoming msg
                        for m in value["messages"]:
                            msg_type = m.get("type")
                            chat.append({
                                "sender": "Employee",
                                "type": msg_type,
                                "text": m.get("text", {}).get("body") if msg_type == "text" else None,
                                "filename": m.get(msg_type, {}).get("filename") if msg_type in ["document", "image", "audio"] else None,
                                "timestamp": m.get("timestamp")
                            })

    chat = sorted(chat, key=lambda x: x.get("timestamp") or "")
    return {"employee": employee, "messages": chat}


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
    append_jsonl(EVENTS_FILE, record)
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
    append_jsonl(EVENTS_FILE, record)
    return {"status": "ok", "file": file.filename, "path": str(file_path)}
