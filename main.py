# --- ElevenLabs API Integration ---
from dashboard_api import find_employee_name_by_msisdn
from elevenlabs_api import tts_to_mp3 , stt_from_mp3
# WhatsApp Lead Agent Bot (Single File)
#
# SETUP NOTES:
# 1. .env file me yeh values daalein:
#    WA_TOKEN=EAAG...                 # WhatsApp access token
#    WA_PHONE_ID=728827243646756      # aapka WA phone number ID
#    BOSS_WA_ID=9232XXXXXXXXX         # boss ka WhatsApp number, country code ke sath, '+' ke bina
#    VERIFY_TOKEN=abc                 # jo aap webhook verify me daloge
#    OPENAI_API_KEY=sk-...            # OpenAI key
#    FOLLOWUP_MINUTES=30              # default one-time follow-up
#    GOOGLE_API_KEY=AIza...           # Gemini API key
#    GEMINI_TEXT_MODEL=gemini-1.5-pro # Gemini text model
#    GEMINI_VISION_MODEL=gemini-1.5-pro # Gemini vision model
#    DELIVERY_DEFAULT=auto            # text | voice | auto
#
# 2. employees.json banayein (example):
#    {"Ali": {"msisdn": "923001234567", "pref": "text"}, "Sara": {"msisdn": "923331234567", "pref": "voice"}, "Bilal": {"msisdn": "923451234567", "pref": "auto"}}
#
# 3. Dependencies install karein (uv se):
#    uv add python-dotenv openai pdfminer.six reportlab pillow imageio-ffmpeg pydub audioop-lts requests google-generativeai
#
# 4. WhatsApp App me webhook URL set karein (e.g., with ngrok):
#    Callback URL: https://<your-ngrok-subdomain>.ngrok-free.app/webhook
#    Verify Token: abc (same as .env)
#
# FEATURES:
# - Boss commands: @tasks/@send, reports (text/pdf/voice), employee mapping
# - Employee updates: Done/Delay commands with progress tracking
# - AI analysis: Images, PDFs, voice transcription
# - Delivery preferences: text/voice/auto per employee
# - Voice forwarding: Forward last voice notes to boss
# - Debouncing: Prevents duplicate webhook processing
# - Professional Roman-Urdu messaging


import os
import warnings


# --- Standard Library Imports ---
import json
import http.server
import socketserver
import threading
import base64
from pathlib import Path
from datetime import datetime, timedelta, timezone
import re
from urllib.parse import urlparse, parse_qs

# --- Third-party Imports ---
import requests
from openai import OpenAI
from pdfminer.high_level import extract_text
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from PIL import Image
import google.generativeai as genai
import io
import time
from dotenv import load_dotenv
load_dotenv(override=True)
# --- Global Configuration & Constants ---
PORT = 8000
HOST = "0.0.0.0"

# Professional Roman-Urdu Phrases
PHRASES = {
    "emp_ack": "Update receive ho gaya. Shukriya.",
    "done_ack": "Update note kar liya. Shukriya.",
    "delay_ack": "Delay note kar liya hai."
}

# Environment Variables
WA_TOKEN = os.getenv("WA_TOKEN")
print(WA_TOKEN)
WA_PHONE_ID = os.getenv("WA_PHONE_ID")
BOSS_WA_ID = os.getenv("BOSS_WA_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-1.5-pro")
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-1.5-pro")
DELIVERY_DEFAULT = os.getenv("DELIVERY_DEFAULT", "auto")
try:
    FOLLOWUP_MINUTES = int(os.getenv("FOLLOWUP_MINUTES", 30))
except ValueError:
    FOLLOWUP_MINUTES = 30

# OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Gemini (primary) + GPT-4o (fallback) router ---
class ModelRouter:
    def __init__(self, openai_client):
        self.client = openai_client
        self.google_key = os.getenv("GOOGLE_API_KEY") or ""
        self.gemini_text = os.getenv("GEMINI_TEXT_MODEL", "gemini-1.5-pro")
        self.gemini_vision = os.getenv("GEMINI_VISION_MODEL", "gemini-1.5-pro")
        if self.google_key:
            genai.configure(api_key=self.google_key)

    # ---- Gemini helpers ----
    def _gemini_text(self, prompt: str) -> str:
        if not self.google_key:
            raise RuntimeError("No GOOGLE_API_KEY")
        m = genai.GenerativeModel(self.gemini_text)
        r = m.generate_content(prompt)
        return (getattr(r, "text", "") or "").strip()

    def _gemini_vision(self, prompt: str, image_bytes: bytes, mime="image/jpeg") -> str:
        if not self.google_key:
            raise RuntimeError("No GOOGLE_API_KEY")
        m = genai.GenerativeModel(self.gemini_vision)
        r = m.generate_content([prompt, {"mime_type": mime, "data": image_bytes}])
        return (getattr(r, "text", "") or "").strip()

    # ---- OpenAI helpers (fallbacks) ----
    def _gpt_text(self, sys: str, user: str, model="gpt-4o") -> str:
        r = self.client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            max_tokens=800,
        )
        return r.choices[0].message.content.strip()

    def _gpt_vision(self, sys: str, image_bytes: bytes) -> str:
        data_url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("utf-8")
        r = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role":"system","content":sys},
                {"role":"user","content":[
                    {"type":"text","text":"Image analysis please."},
                    {"type":"image_url","image_url":{"url":data_url}}
                ]}
            ],
            max_tokens=600
        )
        return r.choices[0].message.content.strip()

    # ---- Public API ----
    def summarize_text(self, sys_style: str, user_prompt: str) -> str:
        try:
            return self._gemini_text(f"{sys_style}\n\n{user_prompt}")
        except Exception:
            try:
                return self._gpt_text(sys_style, user_prompt, model="gpt-4o")
            except Exception:
                return "Temporary analysis issue."

    def summarize_pdf_text(self, sys_style: str, pdf_text: str) -> str:
        chunk = pdf_text[:12000]
        try:
            return self._gemini_text(f"{sys_style}\n\n{chunk}")
        except Exception:
            try:
                return self._gpt_text(sys_style, chunk, model="gpt-4o")
            except Exception:
                return "Temporary PDF analysis issue."

    def analyze_image(self, sys_style: str, image_bytes: bytes) -> str:
        try:
            return self._gemini_vision(sys_style, image_bytes)
        except Exception:
            try:
                return self._gpt_vision(sys_style, image_bytes)
            except Exception:
                return "Temporary image analysis issue."

ROUTER = ModelRouter(client)

# Prompts
GLOBAL_STYLE = "Roman Urdu (Karachi), short & professional. Sirf kaam ki baat. Over-explain nahi. Agar info na mile to seedha keh do."
IMG_PROMPT = "Is tasveer ki seedhi analysis Roman-Urdu me bullets (max 10) likho. Agar text/404/UX issues nazar aayen to mention karo. Tone simple."
PDF_PROMPT = "Roman-Urdu me chhoti summary 3 headings me do: Summary / Action Items / Risks. Sirf kaam ki baat."
TEXT_PROMPT = "Boss ke sawal ka seedha jawab 1 2 lines me. Agar info na ho to batao aur ek clear next step suggest karo."
VOICE_PROMPT = """Transcript ko dekho aur short bullets me likho ke banda kya keh raha hai.
Sirf uske content ka analysis karo (kya bola, kya demand thi, kya issue tha).
Reply ya acknowledgement mat likho.
Roman Urdu me 3–5 simple lines likho.."""

# File Paths
STORAGE_PATHS = {
    "employees": Path("./employees.json"),
    "tasks": Path("./tasks.jsonl"),
    "events": Path("./events.jsonl"),
    "reports": Path("./reports/"),
    "reports_out": Path("./reports/out/"),
    "media": Path("./media/"),
    "seen_ids": Path("./.seen_ids.json"),
    "last_voice": Path("./last_voice.json"),
    "voice_arm": Path("./voice_arm.json")
}

# In-memory state
CAPTURE_STATE = {"on": False, "name": None, "buffer": []}
FOLLOWUP_TIMERS = {}
LAST_MEDIA = {"type": None, "id": None, "filename": None, "sender": None}
SEEN_MESSAGE_IDS = set()  # Track seen message IDs for debouncing

# Pending analysis state for boss reply-driven workflows
BOSS_PENDING = {
    "active": False,
    "awaiting_format": False,
    "kind": None,           # image | pdf | document | audio | text
    "employee": None,
    "path": None,
    "filename": None,
    "text": None,           # for text updates
    "transcript": None,     # for audio
    "summary": None,        # prepared analysis summary
    "ts": 0
}
# Employee-side pending analysis (per-employee)
EMP_PENDING = {}  # { "Hafiz": {"active": True, "items": {"text": "...", "images": [Path..], "docs": [Path..], "audio": [Path..]}} }

LAST_INSTRUCTION_PATH = Path("./reports/last_instruction.json")

def update_last_instruction(name):
    """Update last instruction timestamp for employee."""
    try:
        if LAST_INSTRUCTION_PATH.exists():
            data = json.load(LAST_INSTRUCTION_PATH.open("r", encoding="utf-8"))
        else:
            data = {}
        data[name] = now_iso()
        LAST_INSTRUCTION_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

def get_last_instruction(name):
    """Get last instruction timestamp for employee."""
    try:
        if LAST_INSTRUCTION_PATH.exists():
            data = json.load(LAST_INSTRUCTION_PATH.open("r", encoding="utf-8"))
            return data.get(name)
    except Exception:
        pass
    return None

# --- File & Directory Setup ---
def setup_storage():
    """Create necessary files and directories if they don't exist."""
    for path in STORAGE_PATHS.values():
        if path.suffix:  # It's a file
            if not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                if path.suffix == ".json":
                    if path.name == ".seen_ids.json":
                        path.write_text("[]", encoding="utf-8")
                    elif path.name == "last_voice.json":
                        path.write_text("{}", encoding="utf-8")
                    elif path.name == "voice_arm.json":
                        path.write_text("{}", encoding="utf-8")
                    else:
                        path.write_text("{}", encoding="utf-8")
                else:
                    path.touch()
        else:  # It's a directory
            path.mkdir(parents=True, exist_ok=True)
    
    # Load existing seen message IDs
    global SEEN_MESSAGE_IDS
    try:
        with open(STORAGE_PATHS["seen_ids"], "r", encoding="utf-8") as f:
            SEEN_MESSAGE_IDS = set(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        SEEN_MESSAGE_IDS = set()

def save_seen_message_id(msg_id):
    """Save a message ID to the seen set and persist to disk."""
    global SEEN_MESSAGE_IDS
    SEEN_MESSAGE_IDS.add(msg_id)
    try:
        with open(STORAGE_PATHS["seen_ids"], "w", encoding="utf-8") as f:
            json.dump(list(SEEN_MESSAGE_IDS), f, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving seen IDs: {e}")

def is_message_seen(msg_id):
    """Check if a message ID has already been processed."""
    return msg_id in SEEN_MESSAGE_IDS

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def save_last_voice(name, file_path):
    employees = load_employees()
    emp_data = employees.get(name)
    thread_id = emp_data["msisdn"] if emp_data else name
    path = STORAGE_PATHS["reports"] / f"{thread_id}_voice.json"

    data = {"path": str(file_path), "timestamp": now_iso()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_last_voice(name):
    employees = load_employees()
    emp_data = employees.get(name)
    thread_id = emp_data["msisdn"] if emp_data else name
    path = STORAGE_PATHS["reports"] / f"{thread_id}_voice.json"

    if not path.exists():
        return None
    return json.load(open(path, "r", encoding="utf-8")).get("path")

def get_voice_arm_state():
    """Get current voice arm state."""
    try:
        with open(STORAGE_PATHS["voice_arm"], "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"armed": False, "targets": [], "ts": None}

def set_voice_arm_state(armed, targets=None):
    """Set voice arm state."""
    state = {
        "armed": armed,
        "targets": targets or [],
        "ts": int(time.time()) if armed else None
    }
    try:
        with open(STORAGE_PATHS["voice_arm"], "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving voice arm state: {e}")

def parse_name_list(names_str: str) -> list:
    """Parse comma-separated names into a list."""
    if not names_str.strip():
        return []
    
    # Split by comma, trim spaces, filter empty
    names = [name.strip() for name in names_str.split(",") if name.strip()]
    
    # Title case normalization (but keep original for exact matching)
    normalized = []
    for name in names:
        if name:
            normalized.append(name)
    
    return normalized

def get_employee_by_name(name: str, employees: dict) -> tuple:
    """Get employee data by name (case-insensitive)."""
    # First try exact match
    if name in employees:
        return name, employees[name]
    
    # Then try case-insensitive match
    for emp_name, emp_data in employees.items():
        if emp_name.lower() == name.lower():
            return emp_name, emp_data
    
    return None, None

def mask_msisdn(msisdn: str) -> str:
    """Mask MSISDN for display (e.g., 923001234567 -> 923***4567)."""
    if len(msisdn) >= 7:
        return msisdn[:3] + "***" + msisdn[-4:]
    return msisdn

def append_jsonl(path, data):
    data["timestamp"] = now_iso()
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Error writing to {path}: {e}")

def read_jsonl(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def load_employees():
    try:
        with open(STORAGE_PATHS["employees"], "r", encoding="utf-8") as f:
            raw = json.load(f)
            upgraded = {}
            default_pref = os.getenv("DELIVERY_DEFAULT", "auto")
            for name, val in raw.items():
                if isinstance(val, str):
                    upgraded[name] = {"msisdn": val, "pref": default_pref}
                else:
                    val.setdefault("pref", default_pref)
                    upgraded[name] = val
            return upgraded
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def deliver_to_employee(name: str, msg_text: str, override: str = None) -> dict:
    # Load employee data
    emps = load_employees()
    e = emps.get(name)
    
    # Check if the employee exists in the system
    if not e:
        return {"ok": False, "err": f"{name} not in employees.json"}
    
    # Get employee's MSISDN and preferred mode
    msisdn = e["msisdn"]
    mode = (override or e.get("pref") or os.getenv("DELIVERY_DEFAULT","auto")).lower()

    # Voice Mode - Send TTS audio
    if mode == "voice":
        outp = STORAGE_PATHS["media"] / f"tts_{name}_{int(time.time())}.mp3"
        ok = tts_to_file(msg_text, outp)  # Use existing TTS function (text-to-speech)
        
        if ok:
            # Try sending the audio to the employee
            result = wa_send_audio(msisdn, outp, mark_as_voice=True)
            if result["ok"]:
                return {"ok": True, "mode": "voice"}  # Successful audio delivery
            else:
                # Fallback to text if voice sending fails
                whatsapp_send_text(msisdn, msg_text)
                # Notify boss about the fallback
                if BOSS_WA_ID:
                    whatsapp_send_text(BOSS_WA_ID, f"Voice delivery fail {name}. Text bhej diya.")
                return {"ok": True, "mode": "text_fallback"}
        
        # TTS failed, fallback to text
        whatsapp_send_text(msisdn, msg_text)
        return {"ok": True, "mode": "text_fallback"}

    # Text Mode - Directly send the text message
    if mode == "text":
        whatsapp_send_text(msisdn, msg_text)
        return {"ok": True, "mode": "text"}

    # Auto Mode (default to text if no specific preference is set)
    # This is the fallback mode where text is sent automatically
    whatsapp_send_text(msisdn, msg_text)
    return {"ok": True, "mode": "text_auto"}

def update_boss_number(new_boss_id):
    """Update the boss WhatsApp ID at runtime."""
    global BOSS_WA_ID
    BOSS_WA_ID = new_boss_id
    # Also update environment variable for this session
    os.environ["BOSS_WA_ID"] = new_boss_id

def save_task_record(name, title, due_text):
    task_data = {"name": name, "title": title, "due": due_text, "status": "pending"}
    append_jsonl(STORAGE_PATHS["tasks"], task_data)
    append_jsonl(STORAGE_PATHS["events"], {"kind": "TASK_SAVE", "name": name, "title": title, "due": due_text})

def mark_done_in_tasks(name, title, notes, minutes):
    tasks = read_jsonl(STORAGE_PATHS["tasks"])
    task_found = False
    updated_tasks = []
    for task in tasks:
        if task.get("name") == name and task.get("title") == title and task.get("status") == "pending":
            task["status"] = "completed"
            task["notes"] = notes
            task["minutes"] = minutes
            task["completed_at"] = now_iso()
            task_found = True
        updated_tasks.append(task)

    if not task_found:
        # Create a new task if not found (or if already completed)
        new_task = {
            "name": name, "title": title, "status": "completed",
            "notes": notes, "minutes": minutes, "completed_at": now_iso()
        }
        updated_tasks.append(new_task)
        append_jsonl(STORAGE_PATHS["events"], {"kind": "TASK_COMPLETED_NEW", "name": name, "title": title})
    else:
        append_jsonl(STORAGE_PATHS["events"], {"kind": "TASK_COMPLETED_UPDATE", "name": name, "title": title})
    
    # Rewrite the tasks.jsonl file
    # This is a simple approach. For very large files, a more efficient update might be needed.
    with open(STORAGE_PATHS["tasks"], "w", encoding="utf-8") as f:
        for task in updated_tasks:
            # Remove timestamp added by append_jsonl for consistency before rewriting
            task.pop("timestamp", None)
            f.write(json.dumps(task, ensure_ascii=False) + "\n")



def read_report(name, tail_chars=10000):
    employees = load_employees()
    emp_data = employees.get(name)
    thread_id = emp_data["msisdn"] if emp_data else name
    path = STORAGE_PATHS["reports"] / f"{thread_id}.json"

    if not path.exists():
        return ""
    data = json.load(open(path, "r", encoding="utf-8"))
    msgs = [m for m in data["messages"]]
    return json.dumps(msgs[-50:], ensure_ascii=False, indent=2)  # last 50 msgs

VOICE_ANALYSIS_PROMPT = """
Transcript ko dekho aur short bullets me likho ke banda kya keh raha hai.
Sirf uske content ka analysis karo (kya bola, kya demand thi, kya issue tha).
Reply ya acknowledgement mat likho.
Roman Urdu me 3–5 simple lines likho.
"""


from datetime import datetime, timedelta, timezone

def normalize_ts(ts_string: str):
    """
    Convert timestamp string into naive datetime (UTC-based).
    Supports multiple formats: ISO8601, with/without 'Z', with milliseconds.
    Returns None if parsing fails.
    """
    if not ts_string:
        return None

    try:
        # Standard ISO8601
        ts = datetime.fromisoformat(ts_string.replace("Z", "+00:00"))
    except Exception:
        try:
            # Common fallback: trim microseconds if too long
            if "." in ts_string:
                base, frac = ts_string.split(".", 1)
                frac = frac.rstrip("Z")
                if len(frac) > 6:
                    frac = frac[:6]  # max microseconds length
                ts_string = f"{base}.{frac}"
            ts = datetime.fromisoformat(ts_string.replace("Z", "+00:00"))
        except Exception:
            return None

    # Convert to naive UTC
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    else:
        ts = ts.replace(tzinfo=None)

    return ts



def get_full_report_messages(name):
    """
    Employee ki last 12 ghante ki report ke messages wapas karega,
    Boss ke bheje hue messages hata kar.
    """
    employees = load_employees()
    emp_data = employees.get(name)
    thread_id = emp_data["msisdn"] if emp_data else name
    path = STORAGE_PATHS["reports"] / f"{thread_id}.json"

    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cutoff = datetime.now().replace(tzinfo=None) - timedelta(hours=12)
    msgs = [
        m for m in data.get("messages", [])
        if normalize_ts(m.get("ts"))
        and normalize_ts(m["ts"]) >= cutoff
        and m.get("from") != "Boss"
    ]
    return msgs

#  WhatsApp Helper Functions ---
def whatsapp_send_text(to_msisdn: str, text: str , emp_name: str = None):
    token = os.getenv("WA_TOKEN") or ""
    phone_id = os.getenv("WA_PHONE_ID") or ""
    url = f"https://graph.facebook.com/v23.0/{phone_id}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to_msisdn,
        "type": "text",
        "text": {"body": text}
    }
    headers = {"Authorization": f"Bearer {token}"}
    # debug mask
    if token:
        print("WA_TOKEN(use):", token[:8] + "..." + token[-6:])
    else:
        print("WA_TOKEN(use): NONE")

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        try:
            data = resp.json()
        except Exception:
            data = {"status_code": resp.status_code, "text": resp.text}

        
        employees = load_employees()
        if not emp_name:
            for name, info in employees.items():
                if info.get("msisdn") == to_msisdn:
                    emp_name = name
                    break

        record = {
            "at": now_iso(),
            "kind": "WA_SEND",
            "employee": emp_name,
            "msisdn": to_msisdn,
            "to": to_msisdn,
            "payload": {"text": {"body": text}, "type": "text"}
        }
        append_jsonl(STORAGE_PATHS["events"], record)

        if resp.status_code >= 400:
            handle_whatsapp_error(data, to_msisdn)  # ALWAYS dict-like
        return data

    except requests.RequestException as e:
        # network-level error
        data = {"error": {"message": str(e), "code": -1}}
        append_jsonl(STORAGE_PATHS["events"], {
            "at": now_iso(), "kind": "WA_SEND_ERR", "to": to_msisdn, "err": str(e)
        })
        # yahan koi user-msg dubara mat bhejna
        return data

def handle_whatsapp_error(error_info, to_msisdn: str):
    # error_info kabhi dict, kabhi raw string ho sakta hai — normalize:
    if isinstance(error_info, str):
        try:
            error_info = json.loads(error_info)
        except Exception:
            error_info = {"error": {"message": error_info}}

    err = error_info.get("error", {}) or {}
    code = err.get("code")
    sub  = err.get("error_subcode")
    msg  = err.get("message") or "Unknown error"

    append_jsonl(STORAGE_PATHS["events"], {
        "at": now_iso(), "kind": "WA_ERR",
        "code": code, "subcode": sub, "message": msg
    })

    # SPECIFIC HANDLING:
    if code == 190:
        # Token expired/invalid
        # NOTE: Yahan user ko dobara whatsapp_send_text na call karo; sirf boss ko later inform karna ho to safe jagah pe karo
        print("WA ERROR 190: Token invalid/expired. Refresh WA_TOKEN.")
        return

    if code == 131047:
        # 24h session window band
        print("WA ERROR 131047: 24h session khatam. Template chahiye.")
        return

    if code == 10 or code == 200 or code == 294:  # generic permission/limit
        print(f"WA PERMISSION/LIMIT ERROR: {msg}")
        return

    # default
    print(f"WA ERROR: {code} / {sub} → {msg}")

def wa_get_media_url(media_id):
    url = f"https://graph.facebook.com/v23.0/{media_id}"
    headers = {"Authorization": f"Bearer {WA_TOKEN}"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json().get("url")
    except requests.exceptions.RequestException:
        return None

import requests
from pathlib import Path

def wa_get_media_url(media_id):
    url = f"https://graph.facebook.com/v20.0/{media_id}"
    headers = {"Authorization": f"Bearer {WA_TOKEN}"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json().get("url")
    except requests.exceptions.RequestException:
        return None


def run_grouped_analysis_for_report(name, to_msisdn=None):
    to_msisdn = to_msisdn or BOSS_WA_ID
    msgs = get_full_report_messages(name)
    if not msgs:
        whatsapp_send_text(to_msisdn, f"{name} ki report me analyze karne ko kuch nahi mila.")
        return

    buckets = {"text": [], "image": [], "document": [], "audio": []}
    for m in msgs:
        t = (m.get("type") or "").lower()
        if t in buckets:
            buckets[t].append(m)

    findings = []

    # TEXT
    if buckets["text"]:
        raw = "\n".join([m.get("text", "") for m in buckets["text"]])[:12000]
        if raw.strip():
            txt_sum = summarize_text_content(raw)
            findings.append(f"• TEXT:\n{txt_sum}")

    # IMAGES
    for m in buckets["image"]:
        p = m.get("local_path")
        if p and Path(p).exists():
            out = analyze_image_with_ai(Path(p))
            findings.append(f"• IMAGE {Path(p).name}:\n{out}")

    # DOCUMENTS (pdf/docx/txt/csv)
    for m in buckets["document"]:
        p = m.get("local_path")
        if p and Path(p).exists():
            out = analyze_document_file(Path(p))
            findings.append(f"• DOCUMENT {Path(p).name}:\n{out}")

    # AUDIO
    for m in buckets["audio"]:
        p = m.get("local_path")
        if p and Path(p).exists():
            transcript = transcribe_audio(Path(p))
            try:
                summ = ROUTER.summarize_text(VOICE_ANALYSIS_PROMPT, transcript)
            except Exception:
                summ = transcript
            findings.append(f"• AUDIO {Path(p).name}:\n{summ}")

    if findings:
        combined = f"📊 {name} Report Analysis:\n\n" + "\n\n".join(findings)
        for i in range(0, len(combined), 3500):
            whatsapp_send_text(to_msisdn, combined[i:i+3500])
    else:
        whatsapp_send_text(to_msisdn, f"{name} ki report me analysis ke liye kuch useful nahi tha. Analysis krne ke liye kuch nahi mila. Warne meh Kradeta")

def employee_send_combined_analysis(emp_name, to_msisdn):
    """
    Collect and analyze all pending items (text, images, docs, audio) for a given employee,
    then send a combined summary to WhatsApp.
    """
    state = EMP_PENDING.get(emp_name) or {}
    items = state.get("items") or {}

    findings = []

    # TEXT
    txt = items.get("text")
    if txt and txt.strip():
        findings.append(f"• TEXT:\n{summarize_text_content(txt)}")

    # IMAGES
    for p in items.get("images", []):
        p = Path(p)
        if p.exists():
            out = analyze_image_with_ai(p)
            findings.append(f"• IMAGE {p.name}:\n{out}")

    # DOCUMENTS
    for p in items.get("docs", []):
        p = Path(p)
        if p.exists():
            out = analyze_document_file(p)
            findings.append(f"• DOCUMENT {p.name}:\n{out}")

    # AUDIO
    for p in items.get("audio", []):
        p = Path(p)
        if p.exists():
            transcript = transcribe_audio(p)
            try:
                summ = ROUTER.summarize_text(VOICE_ANALYSIS_PROMPT, transcript)
            except Exception:
                summ = transcript
            findings.append(f"• AUDIO {p.name}:\n{summ}")

    # SEND COMBINED
    if findings:
        combined = f"📝 Boss Message Analysis ({emp_name}):\n\n" + "\n\n".join(findings)
        for i in range(0, len(combined), 3500):  # chunking for WA limit
            whatsapp_send_text(to_msisdn, combined[i:i+3500])
    else:
        whatsapp_send_text(to_msisdn, "Analyze karne ko kuch nahi mila.")

    # reset state
    EMP_PENDING.pop(emp_name, None)

def wa_download_media(media_id: str, out_path: Path) -> Path | None:
    """
    Download media file from WhatsApp Cloud API and save to local path.
    Returns Path if successful, else None.
    """
    media_url = wa_get_media_url(media_id)
    if not media_url:
        print(f"[WA] Failed to fetch media URL for media ID: {media_id}")
        return None

    headers = {"Authorization": f"Bearer {WA_TOKEN}"}
    try:
        response = requests.get(media_url, headers=headers, timeout=30)
        response.raise_for_status()

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "wb") as f:
            f.write(response.content)

        print(f"[WA] Media downloaded successfully to {out_path}")
        return out_path

    except requests.exceptions.RequestException as e:
        print(f"[WA] Error downloading media (ID: {media_id}): {e}")
        return None

def wa_upload_audio(file_path: Path) -> str | None:
    """Upload audio file to WhatsApp Cloud API and return media ID."""
    url = f"https://graph.facebook.com/v23.0/{WA_PHONE_ID}/media"
    headers = {"Authorization": f"Bearer {WA_TOKEN}"}
    
    try:
        with open(file_path, 'rb') as audio_file:
            files = {
                'file': (file_path.name, audio_file, detect_audio_mime(file_path))
            }
            data = {"messaging_product": "whatsapp"}
            
            response = requests.post(url, headers=headers, files=files, data=data, timeout=30)
            
            if response.status_code >= 400:
                # Log error details
                error_data = {
                    "kind": "WA_MEDIA_ERR",
                    "file": str(file_path),
                    "status": response.status_code,
                    "response": response.text,
                    "timestamp": now_iso()
                }
                try:
                    error_data["response_json"] = response.json()
                except:
                    pass
                append_jsonl(STORAGE_PATHS["events"], error_data)
                
                print(f"Audio upload failed: {response.status_code} - {response.text}")
                return None
            
            result = response.json()
            return result.get("id")
            
    except Exception as e:
        error_data = {
            "kind": "WA_MEDIA_ERR", 
            "file": str(file_path),
            "error": str(e),
            "timestamp": now_iso()
        }
        append_jsonl(STORAGE_PATHS["events"], error_data)
        print(f"Audio upload exception: {e}")
        return None

import requests

import mimetypes
import requests
from pathlib import Path

def wa_upload_media(file_path: Path) -> dict:
    """
    Upload local file to WhatsApp Cloud API and return media_id.
    Returns dict: {"ok": True, "media_id": "..."} or {"ok": False, "error": "..."}
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return {"ok": False, "error": f"File not found: {file_path}"}

    mime_type, _ = mimetypes.guess_type(str(file_path))
    if not mime_type:
        # Handle common cases manually
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            mime_type = "application/pdf"
        elif ext in [".jpg", ".jpeg"]:
            mime_type = "image/jpeg"
        elif ext == ".png":
            mime_type = "image/png"
        elif ext in [".mp3", ".mpeg"]:
            mime_type = "audio/mpeg"
        elif ext in [".ogg", ".oga"]:
            mime_type = "audio/ogg"
        else:
            mime_type = "application/octet-stream"

    url = f"https://graph.facebook.com/v23.0/{WA_PHONE_ID}/media"
    headers = {"Authorization": f"Bearer {WA_TOKEN}"}
    try:
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, mime_type)}
            data = {"messaging_product": "whatsapp"}

            resp = requests.post(url, headers=headers, files=files, data=data, timeout=30)
            resp.raise_for_status()

            result = resp.json()
            if "id" in result:
                return {"ok": True, "media_id": result["id"]}
            else:
                return {"ok": False, "error": result}

    except requests.exceptions.RequestException as e:
        return {"ok": False, "error": f"Upload error: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"Unexpected error: {e}"}

def wa_send_document(to_msisdn, media_id=None, filename=None, link=None):
    """
    Send a document via WhatsApp API
    Either pass media_id (uploaded) OR link (public URL).
    """
    url = f"https://graph.facebook.com/v20.0/{WA_PHONE_ID}/messages"
    headers = {"Authorization": f"Bearer {WA_TOKEN}", "Content-Type": "application/json"}

    payload = {
        "messaging_product": "whatsapp",
        "to": to_msisdn,
        "type": "document",
        "document": {}
    }

    if media_id:
        payload["document"]["id"] = media_id
    elif link:
        payload["document"]["link"] = link
    else:
        return {"ok": False, "error": "No media_id or link provided"}

    if filename:
        payload["document"]["filename"] = filename

    resp = requests.post(url, headers=headers, json=payload)
    try:
        result = resp.json()
    except Exception:
        return {"ok": False, "error": f"Invalid response: {resp.text}"}

    if "messages" in result:
        return {"ok": True, "result": result}
    else:
        return {"ok": False, "error": result}


def wa_send_audio(to_msisdn: str, media_id: str | None = None, link: str | None = None, mark_as_voice: bool = False) -> dict:
    """
    Send an audio file via WhatsApp Cloud API.
    - Either pass media_id (uploaded with wa_upload_media) OR link (public URL).
    - If mark_as_voice=True, WhatsApp treats it as a voice note.
    Returns dict: {"ok": True, "result": {...}} or {"ok": False, "error": "..."}
    """
    url = f"https://graph.facebook.com/v23.0/{WA_PHONE_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WA_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": to_msisdn,
        "type": "audio",
        "audio": {}
    }

    if media_id:
        payload["audio"]["id"] = media_id
    elif link:
        payload["audio"]["link"] = link
    else:
        return {"ok": False, "error": "No media_id or link provided"}

    if mark_as_voice:
        payload["audio"]["voice"] = True

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()

        if "messages" in result:
            return {"ok": True, "result": result}
        else:
            return {"ok": False, "error": result}

    except requests.exceptions.RequestException as e:
        return {"ok": False, "error": f"Request error: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"Unexpected error: {e}"}

def wa_send_image(to_msisdn, media_id=None, caption=None, link=None):
    """
    Send an image via WhatsApp API.
    Either pass media_id (uploaded with wa_upload_media) OR link (public URL).
    """
    url = f"https://graph.facebook.com/v20.0/{WA_PHONE_ID}/messages"
    headers = {"Authorization": f"Bearer {WA_TOKEN}", "Content-Type": "application/json"}

    payload = {
        "messaging_product": "whatsapp",
        "to": to_msisdn,
        "type": "image",
        "image": {}
    }

    if media_id:
        payload["image"]["id"] = media_id
    elif link:
        payload["image"]["link"] = link
    else:
        return {"ok": False, "error": "No media_id or link provided"}

    if caption:
        payload["image"]["caption"] = caption

    resp = requests.post(url, headers=headers, json=payload)
    try:
        result = resp.json()
    except Exception:
        return {"ok": False, "error": f"Invalid response: {resp.text}"}

    if "messages" in result:
        return {"ok": True, "result": result}
    else:
        return {"ok": False, "error": result}

# --- OpenAI Helper Functions ---
def analyze_image_with_ai(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            if max(img.size) > 1280:
                img.thumbnail((1280, 1280))
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=92)
            img_bytes = buf.getvalue()
            raw = ROUTER.analyze_image(f"{GLOBAL_STYLE}\n{IMG_PROMPT}", img_bytes)
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()][:10]
            return "\n".join(lines) if lines else "No clear findings."
    except Exception as e:
        append_jsonl(STORAGE_PATHS["events"], {"error": "AI_ERR", "details": str(e)})
        return "Image analysis me temporary issue hai."

def analyze_pdf_with_ai(pdf_path):
    try:
        text = extract_text(pdf_path)[:12000]
        if not text.strip():
            return "PDF se text nahi mila."
        return ROUTER.summarize_pdf_text(f"{GLOBAL_STYLE}\n{PDF_PROMPT}", text)
    except Exception as e:
        append_jsonl(STORAGE_PATHS["events"], {"error": "AI_ERR", "details": str(e)})
        return "PDF analysis me temporary issue hai."

# --- Document/Text Helpers for analysis ---

def extract_text_from_docx_file(path: Path) -> str:
    try:
        import zipfile
        import xml.etree.ElementTree as ET
        with zipfile.ZipFile(path) as z:
            with z.open("word/document.xml") as f:
                xml = f.read()
        root = ET.fromstring(xml)
        texts = []
        for t in root.iter():
            if t.tag.endswith('}t') and t.text:
                texts.append(t.text)
        return " ".join(texts)
    except Exception:
        return ""

def summarize_text_content(text: str) -> str:
    """
    Summarize the given text using AI in Roman Urdu, short and professional.
    """
    try:
        return ROUTER.summarize_text(VOICE_ANALYSIS_PROMPT, text)
    except Exception:
        return "Text analysis me temporary issue hai."


def analyze_document_file(file_path: Path) -> str:
    try:
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return analyze_pdf_with_ai(file_path)
        if ext == ".docx":
            content = extract_text_from_docx_file(file_path)
            if not content.strip():
                return "DOCX se text nahi mila."
            return summarize_text_content(content)
        # Fallback: try read as text
        try:
            raw = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            raw = ""
        if raw.strip():
            return summarize_text_content(raw)
        return "Is document se text extract nahi ho saka."
    except Exception as e:
        append_jsonl(STORAGE_PATHS["events"], {"error": "DOC_ANALYSIS_ERR", "details": str(e)})
        return "Document analysis me issue aaya."


def transcribe_audio(audio_path):
    try:
        transcript = stt_from_mp3(audio_path)
        return transcript
    except Exception as e:
        append_jsonl(STORAGE_PATHS["events"], {"error": "AI_ERR", "details": str(e)})
        return "Audio transcription me temporary issue hai."

def detect_audio_mime(file_path: Path) -> str:
    """Detect MIME type for audio files based on extension."""
    ext = file_path.suffix.lower()
    mime_map = {
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg", 
        ".opus": "audio/ogg",
        ".m4a": "audio/mp4",
        ".mp4": "audio/mp4",
        ".aac": "audio/aac"
    }
    return mime_map.get(ext, "audio/mpeg")  # Default to MP3 for generated files


def tts_to_file(text, out_path):
    try:
        result = tts_to_mp3(text, out_path)
        if result and out_path.exists() and out_path.stat().st_size < 16 * 1024 * 1024:
            return out_path
        else:
            print(f"Warning: TTS file too large or not created: {out_path}")
            return None
    except Exception as e:
        append_jsonl(STORAGE_PATHS["events"], {"error": "AI_ERR", "details": str(e)})
        return None
def write_report(name, header, content, sender="Boss"):
    """
    Save chat/messages per-thread (WhatsApp style).
    Supports both text and media entries.
    """
    try:
        # Employee mapping load karo
        employees = load_employees()
        emp_data = employees.get(name)
        thread_id = emp_data["msisdn"] if emp_data else name

        # File path for JSON thread
        path = STORAGE_PATHS["reports"] / f"{thread_id}.json"

        # Existing thread load karo
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"thread_id": thread_id, "name": name, "messages": []}

        # Timestamp (always naive)
        ts = datetime.now().replace(tzinfo=None).isoformat()

        # Agar content dict hai to assume media entry
        if isinstance(content, dict):
            entry = {
                "from": sender,
                "header": header,
                "type": content.get("type", "text"),
                "filename": content.get("filename"),
                "media_id": content.get("media_id"),
                "local_path": content.get("local_path"),
                "ts": ts
            }
        else:
            # Plain text entry
            entry = {
                "from": sender,
                "header": header,
                "type": "text",
                "text": str(content),
                "ts": ts
            }

        # Add entry to thread
        data["messages"].append(entry)

        # Save back
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"Error writing chat report for {name}: {e}")

# --- Core Logic & Command Handling ---
def polish_to_employee(name, body):
    lines = [f"- {line.strip()}" for line in body.splitlines() if line.strip()]
    formatted_body = "\n".join(lines)
    return f" {name},\nNeeche assignment hai:\n{formatted_body}\n\nProgress par yahin update bhej dein.\n— RollingStones Limited"

def build_ai_summary_text(name):
    try:
        report_content = read_report(name)
        if not report_content.strip():
            return f"'{name}' ki report abhi tak submit nahi hui."
        user = "Is report ko 3–4 lines me concise executive summary Roman-Urdu me do. Kaam ki baat, direct."
        return ROUTER.summarize_text(VOICE_ANALYSIS_PROMPT, user + "\n\n" + report_content)
    except Exception as e:
        append_jsonl(STORAGE_PATHS["events"], {"error": "AI_ERR", "details": str(e)})
        return "Report summary banane me temporary issue hai."

def read_full_day_report(name):
    employees = load_employees()
    emp_data = employees.get(name)
    thread_id = emp_data["msisdn"] if emp_data else name
    path = STORAGE_PATHS["reports"] / f"{thread_id}.json"

    if not path.exists():
        return f"{name} ki koi report nahi mili."

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cutoff = datetime.now().replace(tzinfo=None) - timedelta(hours=12)
        msgs = []
        for m in data.get("messages", []):
            ts_val = normalize_ts(m.get("ts"))
            if ts_val and ts_val >= cutoff and m.get("from") != "Boss":
                msgs.append(m)


        if not msgs:
            return f"{name} ki last 12 ghante ki koi report nahi bani."

        # 1) send everything to Boss
        for m in msgs:
            t = (m.get("type") or "").lower()
            if t == "text":
                whatsapp_send_text(BOSS_WA_ID, f"{name}: {m.get('text','')}")
            elif t == "document":
                if m.get("local_path") and Path(m["local_path"]).exists():
                    media = wa_upload_media(m["local_path"])
                    if media.get("ok"):
                        wa_send_document(BOSS_WA_ID, media_id=media["media_id"], filename=m.get("filename"))
                    else:
                        whatsapp_send_text(BOSS_WA_ID, f"{name}: {m.get('filename','')} (upload fail)")
                elif m.get("media_id"):
                    wa_send_document(BOSS_WA_ID, media_id=m["media_id"], filename=m.get("filename"))
                else:
                    whatsapp_send_text(BOSS_WA_ID, f"{name}: {m.get('filename','')} (no file found)")
            elif t == "image":
                wa_send_image(BOSS_WA_ID, media_id=m.get("media_id"), caption=f"{name} ki image")
            elif t == "audio":
                if m.get("media_id"):
                    wa_send_audio(BOSS_WA_ID, media_id=m["media_id"], mark_as_voice=True)
                else:
                    whatsapp_send_text(BOSS_WA_ID, f"{name}: Audio file missing")

        # 2) ask-once analysis (AFTER sending everything)
        BOSS_PENDING.update({
            "active": True,
            "awaiting_format": False,   # full analysis me format na poochna
            "employee": name,
            "kind": "full",             # pura report analyze karna hai
            "ts": int(time.time())
        })
        whatsapp_send_text(BOSS_WA_ID, f"Boss, {name} ki poori report ka analysis chahiye? (yes/no)")

        return f"{name} ki report bhej di gayi (last 12 ghante)."

    except Exception as e:
        print("❌ Report error:", str(e))
        return f"Report read error: {e}"


def make_report_pdf(name, summary_text, image_notes, pdf_notes, out_path):
    doc = SimpleDocTemplate(str(out_path))
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"<b>Report — {name}</b>", styles['h1']))
    story.append(Paragraph(f"<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>", styles['h3']))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("<b>Executive Summary:</b>", styles['h2']))
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    if image_notes:
        story.append(Paragraph("<b>Image Analysis:</b>", styles['h2']))
        for note in image_notes:
            story.append(Paragraph(note, styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))

    if pdf_notes:
        story.append(Paragraph("<b>PDF Analysis:</b>", styles['h2']))
        for note in pdf_notes:
            story.append(Paragraph(note, styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("— RollingStones Bot", styles['Normal']))

    try:
        doc.build(story)
        return out_path
    except Exception as e:
        append_jsonl(STORAGE_PATHS["events"], {"error": "PDF_GEN_ERR", "details": str(e)})
        return None

def make_report_pdf_all(summary_per_employee, out_path):
    doc = SimpleDocTemplate(str(out_path))
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Combined Employee Report</b>", styles['h1']))
    story.append(Paragraph(f"<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>", styles['h3']))
    story.append(Spacer(1, 0.2 * inch))

    for name, summary in summary_per_employee.items():
        story.append(Paragraph(f"<b>Report for {name}:</b>", styles['h2']))
        story.append(Paragraph(summary, styles['Normal']))
        story.append(Spacer(1, 0.1 * inch))
    
    story.append(Paragraph("— RollingStones Bot", styles['Normal']))

    try:
        doc.build(story)
        return out_path
    except Exception as e:
        append_jsonl(STORAGE_PATHS["events"], {"error": "PDF_GEN_ERR_ALL", "details": str(e)})
        return None



def send_pdf_to_boss(file_path, label_name):
    p = Path(file_path) if not isinstance(file_path, Path) else file_path
    if (not p) or (not p.exists()):
        whatsapp_send_text(BOSS_WA_ID, f"Boss, {label_name} report file nahi mili.")
        return False

    upload = wa_upload_media(p)  # {"ok": bool, "media_id": "..."} expected
    if upload and upload.get("ok") and upload.get("media_id"):
        result = wa_send_document(BOSS_WA_ID, media_id=upload["media_id"], filename=p.name)
        if result.get("ok"):
            whatsapp_send_text(BOSS_WA_ID, f"{label_name} report bhej di gayi hai.")
            return True
        else:
            whatsapp_send_text(BOSS_WA_ID, f"{label_name} report send karte waqt koi error aa gaya.")
            return False
    else:
        whatsapp_send_text(BOSS_WA_ID, f"{label_name} report upload nahi ho saki.")
        return False

def send_audio_to_boss(file_path, label_name):
    p = Path(file_path) if not isinstance(file_path, Path) else file_path
    if (not p) or (not p.exists()):
        whatsapp_send_text(BOSS_WA_ID, f"Boss, {label_name} audio file nahi mili.")
        return False

    upload = wa_upload_media(p)
    if upload and upload.get("ok") and upload.get("media_id"):
        result = wa_send_audio(BOSS_WA_ID, media_id=upload["media_id"], mark_as_voice=True)
        if result.get("ok"):
            whatsapp_send_text(BOSS_WA_ID, f"{label_name} audio file bhej di gayi hai.")
            return True
        else:
            whatsapp_send_text(BOSS_WA_ID, f"{label_name} audio file send karte waqt koi error aa gaya.")
            return False
    else:
        whatsapp_send_text(BOSS_WA_ID, f"{label_name} audio file upload nahi ho sakti.")
        return False


def schedule_followup(name, msisdn):
    cancel_followup(name)
    def send_followup():
        whatsapp_send_text(msisdn, f"Salam {name}, apne kaam ka short update bhej dein. Shukriya.")
        del FOLLOWUP_TIMERS[name]
    
    timer = threading.Timer(FOLLOWUP_MINUTES * 60, send_followup)
    FOLLOWUP_TIMERS[name] = timer
    timer.start()

def cancel_followup(name):
    if name in FOLLOWUP_TIMERS:
        FOLLOWUP_TIMERS[name].cancel()
        del FOLLOWUP_TIMERS[name]

def handle_boss_command(text, from_msisdn):
    global CAPTURE_STATE, BOSS_PENDING
    
    # Log boss intent for debugging
    append_jsonl(STORAGE_PATHS["events"], {
        "kind": "BOSS_INTENT", 
        "text": text, 
        "from": from_msisdn,
        "timestamp": now_iso()
    })
    
    # Pending-analysis conversation handling
    try:
        t = (text or "").strip().lower()
        if BOSS_PENDING.get("active"):
            # First stage: run analysis?
                    if not BOSS_PENDING.get("awaiting_format"):
                        if t in {"yes", "y", "haan", "han", "ji", "ok", "analyze", "analysis"}:
                            kind = BOSS_PENDING.get("kind")
                            emp = BOSS_PENDING.get("employee")

                            # ✅ NEW: full-report case
                            if kind == "full" and emp:
                                run_grouped_analysis_for_report(emp, to_msisdn=from_msisdn)
                                # reset state
                                BOSS_PENDING = {
                                    "active": False, "awaiting_format": False, "kind": None,
                                    "employee": None, "path": None, "filename": None,
                                    "text": None, "transcript": None, "summary": None, "ts": 0
                                }
                                return

                            # 🔽 OLD single-file analysis logic
                            pth = BOSS_PENDING.get("path")
                            summary = None
                            transcript = None
                            if kind == "image" and pth and Path(pth).exists():
                                summary = analyze_image_with_ai(Path(pth))
                            elif kind == "pdf" and pth and Path(pth).exists():
                                summary = analyze_pdf_with_ai(Path(pth))
                            elif kind == "document" and pth and Path(pth).exists():
                                summary = analyze_document_file(Path(pth))
                            elif kind == "audio" and pth and Path(pth).exists():
                                transcript = transcribe_audio(Path(pth))
                                try:
                                    summary = ROUTER.summarize_text(f"{GLOBAL_STYLE}\n{VOICE_PROMPT}", transcript)
                                except Exception:
                                    summary = transcript
                            elif kind == "text":
                                summary = summarize_text_content(BOSS_PENDING.get("text") or "")
                            else:
                                summary = "Analysis path missing."

                            BOSS_PENDING["summary"] = summary
                            BOSS_PENDING["transcript"] = transcript
                            BOSS_PENDING["awaiting_format"] = True

                            # Ask for delivery preference
                            if kind == "audio":
                                whatsapp_send_text(
                                    from_msisdn,
                                    "Boss, analysis tayyar hai. Kya main Original Voice, Transcript, ya Summary (Text/PDF/Voice) me deliver karoon?"
                                )
                            else:
                                whatsapp_send_text(
                                    from_msisdn,
                                    "Boss, analysis tayyar hai. Delivery format? Text, PDF ya Voice? Aap 'all' bhi keh sakte hain."
                                )
                            return

                        elif t in {"no", "n", "skip", "nah", "na"}:
                            BOSS_PENDING = {
                                "active": False, "awaiting_format": False, "kind": None,
                                "employee": None, "path": None, "filename": None,
                                "text": None, "transcript": None, "summary": None, "ts": 0
                            }
                            return
                    else:
                        # awaiting format selection
                        kind = BOSS_PENDING.get("kind")
                        pth = BOSS_PENDING.get("path")
                        summary = BOSS_PENDING.get("summary") or ""
                        transcript = BOSS_PENDING.get("transcript")

                        selected_all = any(w in t for w in ["all", "both"])
                        wants_text = selected_all or ("text" in t)
                        wants_pdf = selected_all or ("pdf" in t)
                        wants_voice = selected_all or ("voice" in t)

                        # Special voice options
                        if kind == "audio" and ("original" in t or "orig" in t):
                            if pth and Path(pth).exists():
                                wa_send_audio(from_msisdn, Path(pth), mark_as_voice=True)
                            BOSS_PENDING = {
                                "active": False, "awaiting_format": False, "kind": None,
                                "employee": None, "path": None, "filename": None,
                                "text": None, "transcript": None, "summary": None, "ts": 0
                            }
                            return

                        if kind == "audio" and "transcript" in t:
                            if transcript:
                                for i in range(0, len(transcript), 3500):
                                    whatsapp_send_text(from_msisdn, transcript[i:i+3500])
                            else:
                                whatsapp_send_text(from_msisdn, "Transcript available nahi hai.")
                            BOSS_PENDING = {
                                "active": False, "awaiting_format": False, "kind": None,
                                "employee": None, "path": None, "filename": None,
                                "text": None, "transcript": None, "summary": None, "ts": 0
                            }
                            return

                        delivered = False
                        # Text
                        if wants_text and summary:
                            for i in range(0, len(summary), 3500):
                                whatsapp_send_text(from_msisdn, summary[i:i+3500])
                            delivered = True

                        # PDF
                        if wants_pdf and summary:
                            out_path = STORAGE_PATHS["reports_out"] / f"{BOSS_PENDING.get('employee','Report')}_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                            if make_report_pdf(BOSS_PENDING.get("employee","Report"), summary, [], [], out_path):
                                send_pdf_to_boss(out_path, BOSS_PENDING.get("employee","Report"))
                                delivered = True
                            else:
                                whatsapp_send_text(from_msisdn, "PDF banane me issue aaya.")

                        # Voice (TTS of summary)
                        if wants_voice and summary:
                            tts_path = STORAGE_PATHS["reports_out"] / f"{BOSS_PENDING.get('employee','Report')}_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.mp3"
                            if tts_to_file(summary, tts_path):
                                wa_send_audio(from_msisdn, tts_path, mark_as_voice=True)
                                delivered = True
                            else:
                                whatsapp_send_text(from_msisdn, "Voice summary ban nahi saki.")
                                    

                        return
    except Exception as e:
        print(f"Error in boss command handling: {e}")
        return

    # A. Capture/Assign commands
    if re.match(r'^@tasks\s+(\w+)$', text, re.IGNORECASE):
        match = re.match(r'^@tasks\s+(\w+)$', text, re.IGNORECASE)
        name = match.group(1).capitalize()
        employees = load_employees()
        if name in employees:
            CAPTURE_STATE = {"on": True, "name": name, "buffer": []}
            whatsapp_send_text(from_msisdn, f"Task capture for {name} shuru. Lines bhejein. '@send' se forward karein.")
        else:
            whatsapp_send_text(from_msisdn, f"'{name}' map nahi. Use: map {name} 923XXXXXXXXX")
        return

    # A2. Multi-recipient @tasks
    if re.match(r'^@tasks\s+([^|:]+)$', text, re.IGNORECASE):
        match = re.match(r'^@tasks\s+([^|:]+)$', text, re.IGNORECASE)
        names_str = match.group(1).strip()
        
        names = parse_name_list(names_str)
        employees = load_employees()
        valid_names = []
        invalid_names = []
        
        for name in names:
            emp_name, emp_data = get_employee_by_name(name, employees)
            if emp_data:
                valid_names.append(emp_name)
            else:
                invalid_names.append(name)
        
      
        return

    # A3. @voice command
        # A3. @voice command
    if re.match(r'^@voice\s+([^|:]+)$', text, re.IGNORECASE):
        match = re.match(r'^@voice\s+([^|:]+)$', text, re.IGNORECASE)
        names_str = match.group(1).strip()
        
        names = parse_name_list(names_str)
        employees = load_employees()
        valid_names = []
        invalid_names = []
        
        for name in names:
            emp_name, emp_data = get_employee_by_name(name, employees)
            if emp_data:
                valid_names.append(emp_name)
            else:
                invalid_names.append(name)
        
        if not valid_names:
            whatsapp_send_text(from_msisdn, f"Koi valid names nahi mile. Use: map <Name> 923XXXXXXXXX")
            return

        # Arm state ON
        set_voice_arm_state(True, valid_names)
        whatsapp_send_text(from_msisdn, f"Voice routing ON: {', '.join(valid_names)}.")
        if invalid_names:
            whatsapp_send_text(from_msisdn, f"Skip: {', '.join(invalid_names)} (map nahi mile).")

        # NOTE: Actual audio forwarding hoga jab boss next baar voice note bheje.
        # Us waqt hum use upload karke employees ko bhejenge.
        return

    if CAPTURE_STATE["on"] and re.match(r'^@send$', text, re.IGNORECASE):
        name = CAPTURE_STATE["name"]
        employees = load_employees()
        
        # Check if this is a multi-recipient task
        if CAPTURE_STATE.get("multi") and CAPTURE_STATE.get("targets"):
            targets = CAPTURE_STATE["targets"]
            body = "\n".join(CAPTURE_STATE["buffer"])
            
            sent_names = []
            skip_names = []
            
            for target_name in targets:
                if target_name in employees:
                    polished_message = polish_to_employee(target_name, body)
                    delivery_result = deliver_to_employee(target_name, polished_message)
                    if delivery_result["ok"]:
                        write_report(target_name, "BOSS -> EMPLOYEE (multi-tasks)", body)
                        schedule_followup(target_name, employees[target_name]["msisdn"])
                        sent_names.append(target_name)
                    else:
                        skip_names.append(f"{target_name} (send fail)")
                else:
                    skip_names.append(f"{target_name} (map nahi mila)")
            
            # Send confirmation
            
                
            
            if skip_names:
                whatsapp_send_text(from_msisdn, f"Skip: {', '.join(skip_names)}.")
        else:
            # Single recipient (existing logic)
            if name in employees:
                body = "\n".join(CAPTURE_STATE["buffer"])
                polished_message = polish_to_employee(name, body)
                delivery_result = deliver_to_employee(name, polished_message)
                
                if delivery_result["ok"]:
                    write_report(name, "BOSS -> EMPLOYEE (forwarded instruction)", body)
                    schedule_followup(name, employees[name]["msisdn"])
                    
                else:
                    whatsapp_send_text(from_msisdn, f"Send fail {name}. Token/session ya mapping check karein.")
          
        
        CAPTURE_STATE = {"on": False, "name": None, "buffer": []}
        return

    # A4. Transcribe command
    if re.match(r'^transcribe\s+(\w+)\s+(yes|no)$', text, re.IGNORECASE):
        match = re.match(r'^transcribe\s+(\w+)\s+(yes|no)$', text, re.IGNORECASE)
        name = match.group(1).capitalize()
        should_transcribe = match.group(2).lower() == "yes"
        
        if should_transcribe:
            last_voice_path = get_last_voice(name)
            if last_voice_path and Path(last_voice_path).exists():
                transcript = transcribe_audio(Path(last_voice_path))
                write_report(name, "VOICE -> TRANSCRIPT (boss approved)", transcript)
                whatsapp_send_text(from_msisdn, "Transcript save ho gayi.")
           
      
        return

    if CAPTURE_STATE["on"]:
        CAPTURE_STATE["buffer"].append(text)
        total_lines = len(CAPTURE_STATE["buffer"])
        whatsapp_send_text(from_msisdn, f"Line add ho gayi (total {total_lines}). '@send' bhejein.")
        return

    # B. Quick Assign (single-shot)
    if re.match(r'^assign\s+(\w+)\s*[|:]\s*(.+)$', text, re.IGNORECASE):
        match = re.match(r'^assign\s+(\w+)\s*[|:]\s*(.+)$', text, re.IGNORECASE)
        name = match.group(1).capitalize()
        message = match.group(2).strip()
        employees = load_employees()
        if name in employees:
            polished_message = polish_to_employee(name, message)
            delivery_result = deliver_to_employee(name, polished_message)
            if delivery_result["ok"]:
                write_report(name, "BOSS -> EMPLOYEE (quick assign)", message)
                schedule_followup(name, employees[name]["msisdn"])
                
            else:
                whatsapp_send_text(from_msisdn, f"Send fail {name}. Token/session ya mapping check karein.")
      
        return

    # B2. Multi-recipient Assign
    if re.match(r'^assign\s+([^|:]+)\s*[|:]\s*(.+)$', text, re.IGNORECASE):
        match = re.match(r'^assign\s+([^|:]+)\s*[|:]\s*(.+)$', text, re.IGNORECASE)
        names_str = match.group(1).strip()
        message = match.group(2).strip()
        
        names = parse_name_list(names_str)
        if len(names) == 1:
            # Single name, handle as before
            name = names[0]
            employees = load_employees()
            if name in employees:
                polished_message = polish_to_employee(name, message)
                delivery_result = deliver_to_employee(name, polished_message)
                if delivery_result["ok"]:
                    write_report(name, "BOSS -> EMPLOYEE (quick assign)", message)
                    schedule_followup(name, employees[name]["msisdn"])
                
                else:
                    whatsapp_send_text(from_msisdn, f"Send fail {name}. Token/session ya mapping check karein.")
        
            return
        
        # Multi-recipient
        employees = load_employees()
        sent_names = []
        skip_names = []
        
        for name in names:
            emp_name, emp_data = get_employee_by_name(name, employees)
            if emp_data:
                polished_message = polish_to_employee(emp_name, message)
                delivery_result = deliver_to_employee(emp_name, polished_message)
                if delivery_result["ok"]:
                    write_report(emp_name, "BOSS -> EMPLOYEE (multi-assign)", message)
                    schedule_followup(emp_name, emp_data["msisdn"])
                    sent_names.append(emp_name)
                else:
                    skip_names.append(f"{name} (send fail)")
            else:
                skip_names.append(f"{name} (map nahi mila)")
        
        # Send confirmation
        if sent_names:
            followup_msg = f"Follow-up {FOLLOWUP_MINUTES} min baad (jahan set ho)."
           
        
        if skip_names:
            whatsapp_send_text(from_msisdn, f"Skip: {', '.join(skip_names)}.")
        return

    # B3. Multi-recipient Ask
    if re.match(r'^ask\s+([^|:]+)\s+(.+)$', text, re.IGNORECASE):
        match = re.match(r'^ask\s+([^|:]+)\s+(.+)$', text, re.IGNORECASE)
        names_str = match.group(1).strip()
        question = match.group(2).strip()
        
        names = parse_name_list(names_str)
        employees = load_employees()
        sent_names = []
        skip_names = []
        
        for name in names:
            emp_name, emp_data = get_employee_by_name(name, employees)
            if emp_data:
                message = f"Boss ne poocha: \"{question}\" — Meharbani apna short update bhej dein."
                delivery_result = deliver_to_employee(emp_name, message)
                if delivery_result["ok"]:
                    schedule_followup(emp_name, emp_data["msisdn"])
                    sent_names.append(emp_name)
                else:
                    skip_names.append(f"{name} (send fail)")
            else:
                skip_names.append(f"{name} (map nahi mila)")
        
        # Send confirmation
        if sent_names:
            followup_msg = f"Follow-up {FOLLOWUP_MINUTES} min baad (jahan set ho)."
            
        
        if skip_names:
            whatsapp_send_text(from_msisdn, f"Skip: {', '.join(skip_names)}.")
        return

    # B4. Multi-recipient Status
    if re.match(r'^status\s+([^|:]+)$', text, re.IGNORECASE):
        match = re.match(r'^status\s+([^|:]+)$', text, re.IGNORECASE)
        names_str = match.group(1).strip()
        
        names = parse_name_list(names_str)
        employees = load_employees()
        sent_names = []
        skip_names = []
        
        for name in names:
            emp_name, emp_data = get_employee_by_name(name, employees)
            if emp_data:
                message = "Kaam ka status?"
                delivery_result = deliver_to_employee(emp_name, message)
                if delivery_result["ok"]:
                    schedule_followup(emp_name, emp_data["msisdn"])
                    sent_names.append(emp_name)
                else:
                    skip_names.append(f"{name} (send fail)")
            else:
                skip_names.append(f"{name} (map nahi mila)")
        
        # Send confirmation
        if sent_names:
            followup_msg = f"Follow-up {FOLLOWUP_MINUTES} min baad (jahan set ho)."
            
        
        if skip_names:
            whatsapp_send_text(from_msisdn, f"Skip: {', '.join(skip_names)}.")
        return

    # B5. Multi-recipient Followup
    if re.match(r'^followup\s+([^|:]+)$', text, re.IGNORECASE):
        match = re.match(r'^followup\s+([^|:]+)$', text, re.IGNORECASE)
        names_str = match.group(1).strip()
        
        names = parse_name_list(names_str)
        employees = load_employees()
        sent_names = []
        skip_names = []
        
        for name in names:
            emp_name, emp_data = get_employee_by_name(name, employees)
            if emp_data:
                message = f"Salam {emp_name}, apne kaam ka short update bhej dein. Shukriya."
                delivery_result = deliver_to_employee(emp_name, message)
                if delivery_result["ok"]:
                    schedule_followup(emp_name, emp_data["msisdn"])
                    sent_names.append(emp_name)
                else:
                    skip_names.append(f"{name} (send fail)")
            else:
                skip_names.append(f"{name} (map nahi mila)")
        
        # Send confirmation
        
        return

    # C. Ask / Status Check (natural language)
    ask_patterns = [
        r'^ask\s+(\w+)\s+(.+)$',
        r'^status\s+(\w+)$',
        r'^pooch\s+(\w+)\s+(.+)$',
        r'^(\w+)\s+se\s+pooch[yi]*\s+(.+)$',
        r'^(\w+)\s+ka\s+status\s+pooch[yi]*$',
        r'^(\w+)\s+se\s+pooch[yi]*\s+kaam\s+hua\s+ya\s+nah\??$'
    ]
    
    for pattern in ask_patterns:
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            if len(match.groups()) == 2:
                name = match.group(1).capitalize()
                question = match.group(2).strip()
            else:
                name = match.group(1).capitalize()
                question = "kaam ka status?"
            
            employees = load_employees()
            if name in employees:
                message = f"Boss ne poocha: \"{question}\" — Meharbani apna short update bhej dein."
                delivery_result = deliver_to_employee(name, message)
                if delivery_result["ok"]:
                    schedule_followup(name, employees[name]["msisdn"])
                    whatsapp_send_text(from_msisdn, f"Message {name} ko bhej diya.")
                else:
                    whatsapp_send_text(from_msisdn, f"Send fail {name}. Token/session ya mapping check karein.")
            else:
                whatsapp_send_text(from_msisdn, f"'{name}' map nahi. Use: map {name} 923XXXXXXXXX")
            return

    # D. Contact / Number
    if re.match(r'^(?:number|contact)\s+(\w+)$', text, re.IGNORECASE):
        match = re.match(r'^(?:number|contact)\s+(\w+)$', text, re.IGNORECASE)
        name = match.group(1).capitalize()
        employees = load_employees()
        if name in employees:
            whatsapp_send_text(from_msisdn, f"{name}: {employees[name]['msisdn']}")
        else:
            whatsapp_send_text(from_msisdn, f"'{name}' map nahi. Use: map {name} 923XXXXXXXXX")
        return

    # E. Delivery Preference
    if re.match(r'^pref\s+(\w+)\s+(text|voice|auto)$', text, re.IGNORECASE):
        match = re.match(r'^pref\s+(\w+)\s+(text|voice|auto)$', text, re.IGNORECASE)
        name = match.group(1).capitalize()
        pref = match.group(2).lower()
        
        employees = load_employees()
        if name not in employees:
            whatsapp_send_text(from_msisdn, f"'{name}' map nahi. Use: map {name} 923XXXXXXXXX")
            return
        
        employees[name]["pref"] = pref
        try:
            with open(STORAGE_PATHS["employees"], "w", encoding="utf-8") as f:
                json.dump(employees, f, ensure_ascii=False, indent=2)
            whatsapp_send_text(from_msisdn, f"{name} ki preference '{pref}' set ho gayi.")
        except Exception as e:
            whatsapp_send_text(from_msisdn, f"Preference save karne me issue: {e}")
        return

    # F. Mapping
    if re.match(r'^map\s+(\w+)\s+(\d+)$', text, re.IGNORECASE):
        match = re.match(r'^map\s+(\w+)\s+(\d+)$', text, re.IGNORECASE)
        name = match.group(1).capitalize()
        msisdn = match.group(2).strip()
        employees = load_employees()
        employees[name] = {"msisdn": msisdn, "pref": os.getenv("DELIVERY_DEFAULT", "auto")}
        
        try:
            with open(STORAGE_PATHS["employees"], "w", encoding="utf-8") as f:
                json.dump(employees, f, ensure_ascii=False, indent=2)
            whatsapp_send_text(from_msisdn, f"{name} ko {msisdn} se map kar diya gaya hai.")
        except Exception as e:
            whatsapp_send_text(from_msisdn, f"Mapping save karne me issue: {e}")
        return

    # G. Voice Forward
    if re.match(r'^forward\s+(?:last\s+)?voice\s+(\w+)$', text, re.IGNORECASE):
        match = re.match(r'^forward\s+(?:last\s+)?voice\s+(\w+)$', text, re.IGNORECASE)
        name = match.group(1).capitalize()
        last_voice_path = get_last_voice(name)
        
        if not last_voice_path or not Path(last_voice_path).exists():
            whatsapp_send_text(from_msisdn, f"{name} ke last voice ka record nahi mila.")
            return
        
        voice_file = Path(last_voice_path)
        result = wa_send_audio(from_msisdn, voice_file, mark_as_voice=True)
       
        return

    # H. Reports
       # H. Reports
    # Case 1: report <name> → full daily report (all entries of today)
    if re.match(r'^@working\s+(\w+)\s+(.+)$', text, re.IGNORECASE):
        match = re.match(r'^@working\s+(\w+)\s+(.+)$', text, re.IGNORECASE)
        name = match.group(1).capitalize()
        message = match.group(2).strip()
        employees = load_employees()

        if name not in employees:
            whatsapp_send_text(from_msisdn, f"'{name}' map nahi. Use: map {name} 923XXXXXXXXX")
            return

        # Step 1: Boss ka message employee ko bhejna (exact same message)
        polished_message = f"Boss ka working message:\n\n{message}"
        deliver_to_employee(name, polished_message)
        write_report(name, "BOSS -> EMPLOYEE (working) >>>>>>>    ", message)
        
        # Step 2: Employee se analysis ke liye puchhna
        whatsapp_send_text(employees[name]["msisdn"], f"{name}, boss ne ek working bheji hai. Analysis chahiye? (yes/no)")

        # Step 3: Pending state for boss's request
        BOSS_PENDING.update({
            "active": True,
            "awaiting_format": False,
            "kind": "text",
            "employee": name,
            "text": message,
            "summary": None,
            "ts": int(time.time())
        })
        
        # Step 4: Handling long messages and sending them in chunks
        max_chunk_size = 3500  # WhatsApp's character limit per message
        for i in range(0, len(message), max_chunk_size):
            chunk = message[i:i + max_chunk_size]
            whatsapp_send_text(from_msisdn, chunk)  # Send chunk to the employee in parts

        return

    if re.match(r'^report\s+(\w+)$', text, re.IGNORECASE):
        match = re.match(r'^report\s+(\w+)$', text, re.IGNORECASE)
        name = match.group(1).capitalize()
        employees = load_employees()

        if name not in employees:
                       whatsapp_send_text(from_msisdn, f"'{name}' map nahi. Use: map {name} 923XXXXXXXXX")
      

        # Get the full day's report ...
        # Get and forward the last 12h raw items to Boss (inside read_full_day_report)
        status = read_full_day_report(name)
        whatsapp_send_text(from_msisdn, status)
        return



    # Case 2: report <name> text/pdf/voice → AI summary formats
    if re.match(r'^report\s+(\w+)\s+(text|pdf|voice)$', text, re.IGNORECASE):
        match = re.match(r'^report\s+(\w+)\s+(text|pdf|voice)$', text, re.IGNORECASE)
        name = match.group(1).capitalize()
        report_format = match.group(2).lower()
        
        employees = load_employees()
        if name not in employees:
            whatsapp_send_text(from_msisdn, f"'{name}' map nahi. Use: map {name} 923XXXXXXXXX")
            return

        if report_format == "text":
            summary = build_ai_summary_text(name)
            whatsapp_send_text(from_msisdn, summary)
        elif report_format == "pdf":
            summary = build_ai_summary_text(name)
            report_content = read_report(name)
            image_notes = re.findall(r"IMAGE -> ANALYSIS\n(.*?)(?=\n---|\Z)", report_content, re.DOTALL)[-3:]
            pdf_notes = re.findall(r"PDF -> ANALYSIS:.*?\n(.*?)(?=\n---|\Z)", report_content, re.DOTALL)[-3:]
            
            out_path = STORAGE_PATHS["reports_out"] / f"{name}_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            if make_report_pdf(name, summary, image_notes, pdf_notes, out_path):
                send_pdf_to_boss(out_path, name)
            else:
                whatsapp_send_text(from_msisdn, f"{name} ki PDF report banane me issue hai.")
        elif report_format == "voice":
            summary = build_ai_summary_text(name)
            tts_path = STORAGE_PATHS["reports_out"] / f"{name}_report_{datetime.now().strftime('%Y%m%d_%H%M')}.mp3"
            if tts_to_file(summary, tts_path):
                result = wa_send_audio(from_msisdn, tts_path, mark_as_voice=False)
               
                
        return
    
    if re.match(r'^report\s+all\s+(text|pdf)$', text, re.IGNORECASE):
        match = re.match(r'^report\s+all\s+(text|pdf)$', text, re.IGNORECASE)
        report_format = match.group(1).lower()
        employees = load_employees()
        
        if report_format == "text":
            all_summaries = {}
            for name in employees.keys():
                summary = build_ai_summary_text(name)
                all_summaries[name] = summary
            
            combined_text = "\n\n".join([f"--- {name} ---\n{summary}" for name, summary in all_summaries.items()])
            for i in range(0, len(combined_text), 3500):
                whatsapp_send_text(from_msisdn, combined_text[i:i+3500])
           
        elif report_format == "pdf":
            all_summaries = {}
            for name in employees.keys():
                summary = build_ai_summary_text(name)
                all_summaries[name] = summary
            
            out_path = STORAGE_PATHS["reports_out"] / f"combined_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            if make_report_pdf_all(all_summaries, out_path):
                send_pdf_to_boss(out_path, "Combined")
            else:
                whatsapp_send_text(from_msisdn, "Combined PDF report banane me issue hai.")
        return


    # I. Boss Identity
    if re.match(r'^setboss\s+(\d+)$', text, re.IGNORECASE):
        match = re.match(r'^setboss\s+(\d+)$', text, re.IGNORECASE)
        new_boss_id = match.group(1).strip()
        update_boss_number(new_boss_id)
        whatsapp_send_text(from_msisdn, f"Boss number set ho gaya: {new_boss_id}. Ab aap commands use kar sakte hain.")
        whatsapp_send_text(new_boss_id, "Aap ko boss ke tor par confirm kar diya gaya hai.")
        return

    # J. Docs command
    if re.match(r'^docs?$', text, re.IGNORECASE):
        docs_text = """Bot Use – Shortcut

assign Hira, Ali | Banner fix
ask Hira Kaam hua?
@tasks Ali … @send
@voice Hira (then send audio)
report Ali pdf|text|voice
employees, pref Ali voice

Agar voice aayegi to aap se poochun ga: transcript chahiye?"""
        whatsapp_send_text(from_msisdn, docs_text)
        return

    # K. Employees list
    if re.match(r'^employees?$', text, re.IGNORECASE):
        employees = load_employees()
        if not employees:
            whatsapp_send_text(from_msisdn, "Koi employees map nahi hain. Use: map <Name> <msisdn>")
            return
        
        employee_list = []
        for name, data in employees.items():
            masked_msisdn = mask_msisdn(data["msisdn"])
            pref = data.get("pref", "auto")
            employee_list.append(f"{name} — {masked_msisdn} — pref: {pref}")
        
        response = "Employees:\n" + "\n".join(employee_list)
        whatsapp_send_text(from_msisdn, response)
        return


    m = re.match(r'^analyze\s+(\w+)$', text, re.IGNORECASE)
    if m:
        emp = m.group(1).capitalize()
        run_grouped_analysis_for_report(emp, to_msisdn=from_msisdn)
        BOSS_PENDING = {"active": False, "awaiting_format": False, "kind": None, "employee": None,
                        "path": None, "filename": None, "text": None, "transcript": None, "summary": None, "ts": 0}
        return

    # L. Followup command
   
    # Case 1: report <name> → full daily report (all entries of today)
    if re.match(r'^@working\s+(\w+)\s+(.+)$', text, re.IGNORECASE):
        match = re.match(r'^@working\s+(\w+)\s+(.+)$', text, re.IGNORECASE)
        name = match.group(1).capitalize()
        message = match.group(2).strip()
        employees = load_employees()

        if name not in employees:
            whatsapp_send_text(from_msisdn, f"'{name}' map nahi. Use: map {name} 923XXXXXXXXX")
            return

        # Step 1: Boss ka message employee ko bhejna (exact same message)
        polished_message = f"Boss ka working message:\n\n{message}"
        deliver_to_employee(name, polished_message)
        write_report(name, "BOSS -> EMPLOYEE (working)", message)
        
        # Step 2: Employee se analysis ke liye puchhna
       # Step 2: Employee se analysis ke liye puchhna
        whatsapp_send_text(employees[name]["msisdn"], f"{name}, Boss ne working bheji hai. Analysis chahiye? (yes/no)")

        # Step 3: Pending state for EMPLOYEE
        EMP_PENDING[name] = {
            "active": True,
            "items": {"text": message, "images": [], "docs": [], "audio": []},
            "ts": int(time.time())
        }


        # Step 4: Handling long messages and sending them in chunks
        max_chunk_size = 3500  # WhatsApp's character limit per message
        for i in range(0, len(message), max_chunk_size):
            chunk = message[i:i + max_chunk_size]
            whatsapp_send_text(from_msisdn, chunk)  # Send chunk to the employee in parts

        return

def handle_employee_message(text, from_msisdn, sender_name):
    
    global LAST_MEDIA, BOSS_PENDING
    tnorm = (text or "").strip().lower()
    if EMP_PENDING.get(sender_name, {}).get("active"):
        if tnorm in {"yes", "y", "haan", "han", "ji", "ok", "analyze", "analysis"}:
            employee_send_combined_analysis(sender_name, from_msisdn)
            return
        elif tnorm in {"no", "n", "skip", "nah", "na"}:
            EMP_PENDING.pop(sender_name, None)
            whatsapp_send_text(from_msisdn, "Theek hai, analysis skip kar diya.")
            return
    if text.lower().startswith("done"):
        # Handle "done" message
        write_report(sender_name, "EMPLOYEE -> UPDATE", text , sender="Employee")
    
        
        # Check if there are any media to send
        if LAST_MEDIA.get("type"):
            media_messages = []
            if LAST_MEDIA.get("audio"):
                media_messages.append(LAST_MEDIA["audio"])
            if LAST_MEDIA.get("images"):
                media_messages.extend(LAST_MEDIA["images"])
            if LAST_MEDIA.get("pdf"):
                media_messages.append(LAST_MEDIA["pdf"])
            # Send all media in one message
            if media_messages:
                for media in media_messages:
                    if media["type"] == "audio":
                        wa_send_audio(BOSS_WA_ID, media["id"])
                    elif media["type"] == "image":
                        wa_send_image(BOSS_WA_ID, media["id"])
                    elif media["type"] == "pdf":
                        wa_send_document(BOSS_WA_ID, media["id"], media["filename"])
                whatsapp_send_text(BOSS_WA_ID, f"Boss, {sender_name} ke {len(media_messages)} items forward ho gaye hain. Apko short analysis chahiye?")
        return
    elif text.lower().startswith("delay"):
        # Handle "delay" message
        write_report(sender_name, "EMPLOYEE -> DELAY", text)
        
       
        return
    # General update
    write_report(sender_name, "EMPLOYEE -> UPDATE", text , sender="Employee")
    
    
    # Offer analysis to boss for text updates
    

def handle_media(media_type, media_id, from_msisdn, sender_name, mime_type=None, filename=None):
    global LAST_MEDIA
    date_str = datetime.now().strftime("%Y-%m-%d")
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    media_dir = STORAGE_PATHS["media"] / sender_name / date_str
    media_dir.mkdir(parents=True, exist_ok=True)

    ack_msg, file_path = None, None

    # --- IMAGE ---
    if media_type == "image":
        file_path = wa_download_media(media_id, media_dir / f"image_{ts_str}.jpg")
        ack_msg = "Image receive ho gayi."

        if file_path:
            # Upload to WA cloud
            res = wa_upload_media(file_path)
            if res["ok"]:
                mid = res["media_id"]
                write_report(sender_name, "EMPLOYEE -> BOSS (image)", filename or file_path.name,
                             sender="Employee")
                # Save in JSON thread with media_id
                write_report(sender_name, "EMPLOYEE_MEDIA", {
                    "type": "image",
                    "filename": file_path.name,
                    "media_id": mid,
                    "local_path": str(file_path)
                }, sender="Employee")
                LAST_MEDIA = {"type": "image", "filename": file_path.name,
                              "sender": sender_name, "media_id": mid,
                              "local_path": str(file_path)}

    # --- DOCUMENT / PDF ---
    elif media_type == "document":
        ext = Path(filename).suffix if filename else ".bin"
        file_path = wa_download_media(media_id, media_dir / f"doc_{ts_str}{ext}")
        ack_msg = "Document receive ho gaya."

        if file_path:
            res = wa_upload_media(file_path)
            if res["ok"]:
                mid = res["media_id"]
                write_report(sender_name, "EMPLOYEE -> BOSS (document)", filename or file_path.name,
                             sender="Employee")
                write_report(sender_name, "EMPLOYEE_MEDIA", {
                    "type": "document",
                    "filename": file_path.name,
                    "media_id": mid,
                    "local_path": str(file_path)
                }, sender="Employee")
                LAST_MEDIA = {"type": "document", "filename": file_path.name,
                              "sender": sender_name, "media_id": mid,
                              "local_path": str(file_path)}

    # --- AUDIO / VOICE ---
    elif media_type == "audio":
        file_path = wa_download_media(media_id, media_dir / f"audio_{ts_str}.ogg")
        ack_msg = "Voice note receive ho gaya."

        if file_path:
            res = wa_upload_media(file_path)
            if res["ok"]:
                mid = res["media_id"]
                save_last_voice(sender_name, file_path)
                write_report(sender_name, "EMPLOYEE -> BOSS (audio)", file_path.name, sender="Employee")
                write_report(sender_name, "EMPLOYEE_MEDIA", {
                    "type": "audio",
                    "filename": file_path.name,
                    "media_id": mid,
                    "local_path": str(file_path)
                }, sender="Employee")
                LAST_MEDIA = {"type": "audio", "filename": file_path.name,
                              "sender": sender_name, "media_id": mid,
                              "local_path": str(file_path)}

    # Send ack to employee only
    if ack_msg:
        whatsapp_send_text(from_msisdn, ack_msg)

# --- HTTP Server ---
class WebhookHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        query = parse_qs(parsed_path.query)
        
        if parsed_path.path == "/webhook":
            if ('hub.mode' in query and 'hub.verify_token' in query and
                    query['hub.mode'][0] == 'subscribe' and
                    query['hub.verify_token'][0] == VERIFY_TOKEN):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(query['hub.challenge'][0].encode())
            else:
                self.send_response(403)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/webhook":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)

            # extract wa_id from incoming message
            wa_id = None
            emp_name = None
            if "entry" in data:
                for entry in data["entry"]:
                    for change in entry.get("changes", []):
                        value = change.get("value", {})
                        contacts = value.get("contacts", [])
                        for c in contacts:
                            wa_id = c.get("wa_id")
                            emp_name = find_employee_name_by_msisdn(wa_id)

            # log event
            append_jsonl(STORAGE_PATHS["events"], {
                "kind": "WA_RECV",
                "employee": emp_name,
                "msisdn": wa_id,
                "payload": data,
                "timestamp": now_iso()
            })

            # normal processing of messages
            if 'entry' in data and data['entry']:
                for entry in data['entry']:
                    for change in entry.get('changes', []):
                        if 'messages' in change.get('value', {}):
                            for message in change['value']['messages']:
                                self.process_message(message)

            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def process_message(self, message):
        from_msisdn = message.get('from')
        msg_type = message.get('type')
        msg_id = message.get('id')

        # Debug logging
        print(f"Message from {from_msisdn}, type: {msg_type}")

        # Check for duplicate messages
        if msg_id and is_message_seen(msg_id):
            print("Duplicate message detected, skipping")
            return
        
        # Mark message as seen
        if msg_id:
            save_seen_message_id(msg_id)

        employees = load_employees()
        sender_name = None
        sender_info = None
        for name, info in employees.items():
            if info["msisdn"] == from_msisdn:
                sender_name = name
                sender_info = info
                break
        is_boss = (from_msisdn == BOSS_WA_ID)

        if not is_boss and sender_name:
            # Always notify boss with mapped name and number
            
            # Confirm to employee
            whatsapp_send_text(from_msisdn, f" {sender_name}, aapka message receive ho gaya hai..")
            # Proceed with employee message handling
            if msg_type == 'text':
                text = message['text']['body']
                handle_employee_message(text, from_msisdn, sender_name)
            elif msg_type in ['image', 'document', 'audio', 'voice']:
                if msg_type == 'voice':
                    msg_type = 'audio'
                medi_id = message[msg_type]['id']
                mime_type = message[msg_type].get('mime_type')
                filename = message[msg_type].get('filename')
                handle_media(msg_type, medi_id, from_msisdn, sender_name, mime_type, filename)
            return

        if not is_boss and not sender_name:
            # Unknown sender, ask for clarification
            whatsapp_send_text(BOSS_WA_ID, f"Unknown sender {from_msisdn}. Please map this number to a name for best experience.")
            whatsapp_send_text(from_msisdn, "Aapka number system me map nahi hai. Meharbani se apna naam confirm karein.")
            return

        if is_boss:
            if msg_type == 'text':
                text = message['text']['body']
                handle_boss_command(text, from_msisdn)
                voice_arm_state = get_voice_arm_state()
                if voice_arm_state.get("armed"):
                    set_voice_arm_state(False)
            elif msg_type in ['image', 'document', 'audio', 'voice']:
                if msg_type == 'voice':
                    msg_type = 'audio'
                medi_id = message[msg_type]['id']
                mime_type = message[msg_type].get('mime_type')
                filename = message[msg_type].get('filename')
                handle_media(msg_type, medi_id, from_msisdn, "Boss", mime_type, filename)
        elif msg_type in ['image', 'document', 'audio', 'voice']:
            # Normalize voice/audio types
            if msg_type == 'voice':
                msg_type = 'audio'  # Treat voice as audio for processing
            
            medi_id = message[msg_type]['id']
            mime_type = message[msg_type].get('mime_type')
            filename = message[msg_type].get('filename')
            handle_media(msg_type, medi_id, from_msisdn, sender_name, mime_type, filename)
            
            # Check if boss sent audio and voice arm is active
            if is_boss and msg_type == 'audio':
                voice_arm_state = get_voice_arm_state()
                if voice_arm_state.get("armed") and voice_arm_state.get("targets"):
                    # Download the audio first
                    date_str = datetime.now().strftime("%Y-%m-%d")
                    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    media_dir = STORAGE_PATHS["media"] / "boss" / date_str
                    file_path = wa_download_media(medi_id, media_dir / f"boss_audio_{ts_str}.ogg")
                    
                    if file_path:
                        targets = voice_arm_state["targets"]
                        sent_names = []
                        skip_names = []
                        
                        for target_name in targets:
                            if target_name in employees:
                                result = wa_send_audio(employees[target_name]["msisdn"], file_path, mark_as_voice=True)
                                if result["ok"]:
                                    sent_names.append(target_name)
                                else:
                                    skip_names.append(f"{target_name} (send fail)")
                            else:
                                skip_names.append(f"{target_name} (map nahi mila)")
                        
                        # Send confirmation and disarm
                        if sent_names:
                            whatsapp_send_text(from_msisdn, f"Voice routing OFF. Bhej diya: {', '.join(sent_names)}.")
                        
                        if skip_names:
                            whatsapp_send_text(from_msisdn, f"Skip: {', '.join(skip_names)}.")
                        
                        set_voice_arm_state(False)
                    else:
                        whatsapp_send_text(from_msisdn, "Voice routing fail. Audio download nahi hua.")
                        set_voice_arm_state(False)



def main():
    global BOSS_WA_ID
    
    if not all([WA_TOKEN, WA_PHONE_ID, BOSS_WA_ID]):
        warnings.warn("Warning: One or more required environment variables (WA_TOKEN, WA_PHONE_ID, BOSS_WA_ID) are missing.")
    
    setup_storage()
    
    print(f"Boss WhatsApp ID: {BOSS_WA_ID}")
    print("New commands: setboss, map, pref, forward voice, report <Name> voice")
    print("Multi-recipient: assign Hira,Ali | message, ask Hira,Ali question, status Hira,Ali")
    print("Voice routing: @voice Hira,Ali (then send audio), transcribe <Name> yes/no")
    print("Docs: docs, employees")
    print("NOTE: Dev mode me sirf test numbers ko messages jaate hain. WhatsApp App → Phone numbers → Add tester.")
    
    with socketserver.TCPServer((HOST, PORT), WebhookHandler) as httpd:
        print(f"Listening on http://{HOST}:{PORT} (POST/GET /webhook)")
        httpd.serve_forever()

if __name__ == "__main__":
    main()


