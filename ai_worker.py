import os, traceback, base64, json, pickle
from pathlib import Path
from datetime import datetime
import google.generativeai as genai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import psycopg2
from psycopg2.extras import execute_values
from filelock import FileLock

load_dotenv()

# 🔹 Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# 🔹 Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# GPT model (text replies)
gpt_model = ChatOpenAI(model="gpt-4o-mini",
                       temperature=0.7,
                       openai_api_key=OPENAI_API_KEY)

# OpenAI client for vision API and Whisper STT
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ============================================================
# Per-Employee Memory Store (JSONL-based + PostgreSQL + FAISS)
# ============================================================

MEMORY_DIR = Path("./memory")
MEMORY_DIR.mkdir(exist_ok=True)

# FAISS vector store directory
FAISS_DIR = Path("./data")
FAISS_DIR.mkdir(exist_ok=True)

# OpenAI Embeddings for FAISS
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
) if OPENAI_API_KEY else None


class EmployeeChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in JSONL file per employee."""
    
    def __init__(self, employee_id: str, max_messages: int = 20):
        self.employee_id = employee_id
        self.max_messages = max_messages
        self.file_path = MEMORY_DIR / f"{employee_id}.jsonl"
        self._messages = self._load_messages()
    
    def _load_messages(self):
        """Load last N messages from JSONL file."""
        if not self.file_path.exists():
            return []
        
        messages = []
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                # Get last max_messages entries
                for line in lines[-self.max_messages:]:
                    record = json.loads(line.strip())
                    if record.get("role") == "human":
                        messages.append(HumanMessage(content=record["content"]))
                    elif record.get("role") == "ai":
                        messages.append(AIMessage(content=record["content"]))
        except Exception as e:
            print(f"❌ Error loading memory for {self.employee_id}: {e}")
        
        return messages
    
    def add_message(self, message):
        """Add a message to history and persist to file."""
        self._messages.append(message)
        
        # Persist to JSONL
        try:
            with open(self.file_path, "a", encoding="utf-8") as f:
                record = {
                    "at": datetime.utcnow().isoformat(),
                    "role": "human" if isinstance(message, HumanMessage) else "ai",
                    "content": message.content
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"❌ Error saving message for {self.employee_id}: {e}")
    
    @property
    def messages(self):
        return self._messages
    
    def clear(self):
        """Clear message history."""
        self._messages = []
        if self.file_path.exists():
            self.file_path.unlink()


# Memory store for all employees
employee_memory_store = {}


def get_employee_memory(employee_id: str) -> BaseChatMessageHistory:
    """Get or create memory for an employee."""
    if employee_id not in employee_memory_store:
        employee_memory_store[employee_id] = EmployeeChatMessageHistory(employee_id)
    return employee_memory_store[employee_id]


# ============================================================
# PostgreSQL Persistent Memory Functions
# ============================================================

def get_db_connection():
    """Get PostgreSQL database connection."""
    try:
        if not DATABASE_URL:
            print("⚠️ DATABASE_URL not configured")
            return None
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return None


def save_memory_to_postgres(staff_id: str, context: str, embedding_bytes: bytes = None):
    """Save conversation context to PostgreSQL staff_memory table."""
    try:
        conn = get_db_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        # Upsert: update if exists, insert if not
        cursor.execute("""
            INSERT INTO staff_memory (staff_id, context, embedding, updated_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (staff_id)
            DO UPDATE SET 
                context = EXCLUDED.context,
                embedding = EXCLUDED.embedding,
                updated_at = NOW()
        """, (staff_id, context, embedding_bytes))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        log_memory_event(staff_id, "memory_sync", "PostgreSQL save successful")
        return True
        
    except Exception as e:
        print(f"❌ PostgreSQL save error for {staff_id}: {e}")
        traceback.print_exc()
        return False


def load_memory_from_postgres(staff_id: str) -> str:
    """Load conversation context from PostgreSQL."""
    try:
        conn = get_db_connection()
        if not conn:
            return None
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT context FROM staff_memory WHERE staff_id = %s
        """, (staff_id,))
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if result:
            return result[0]
        return None
        
    except Exception as e:
        print(f"❌ PostgreSQL load error for {staff_id}: {e}")
        return None


def get_or_create_memory(staff_id: str):
    """
    Unified memory retrieval: checks PostgreSQL, then JSONL, then creates new.
    Returns both JSONL-based chat history and PostgreSQL context.
    """
    # Get JSONL-based message history (for LangChain)
    chat_history = get_employee_memory(staff_id)
    
    # Get PostgreSQL context (for persistence)
    postgres_context = load_memory_from_postgres(staff_id)
    
    return chat_history, postgres_context


# ============================================================
# FAISS Vector Store Functions
# ============================================================

# Cache for FAISS vector stores per staff
faiss_stores = {}


def get_faiss_store(staff_id: str):
    """Get or create FAISS vector store for staff member."""
    try:
        if staff_id in faiss_stores:
            return faiss_stores[staff_id]
        
        faiss_path = FAISS_DIR / f"faiss_{staff_id}"
        
        # Load existing index if available
        if faiss_path.exists() and embeddings_model:
            try:
                vectorstore = FAISS.load_local(
                    str(faiss_path),
                    embeddings_model,
                    allow_dangerous_deserialization=True
                )
                faiss_stores[staff_id] = vectorstore
                print(f"✅ Loaded FAISS index for {staff_id}")
                return vectorstore
            except Exception as e:
                print(f"⚠️ Error loading FAISS for {staff_id}: {e}")
        
        # Create new empty store
        if embeddings_model:
            vectorstore = FAISS.from_texts(
                ["Initial context for " + staff_id],
                embeddings_model,
                metadatas=[{"staff_id": staff_id, "type": "init"}]
            )
            faiss_stores[staff_id] = vectorstore
            print(f"✅ Created new FAISS index for {staff_id}")
            return vectorstore
        
        return None
        
    except Exception as e:
        print(f"❌ FAISS get/create error for {staff_id}: {e}")
        traceback.print_exc()
        return None


def embed_text_with_faiss(staff_id: str, text: str, role: str = "human"):
    """
    Embed and store text in FAISS vector store.
    Persists index to disk after update.
    """
    try:
        if not embeddings_model:
            print("⚠️ Embeddings model not configured")
            return False
        
        vectorstore = get_faiss_store(staff_id)
        if not vectorstore:
            return False
        
        # Add text with metadata
        metadata = {
            "staff_id": staff_id,
            "role": role,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        vectorstore.add_texts([text], metadatas=[metadata])
        
        # Persist to disk
        faiss_path = FAISS_DIR / f"faiss_{staff_id}"
        vectorstore.save_local(str(faiss_path))
        
        log_memory_event(staff_id, "faiss_embed", f"Embedded {role} message ({len(text)} chars)")
        return True
        
    except Exception as e:
        print(f"❌ FAISS embed error for {staff_id}: {e}")
        traceback.print_exc()
        return False


def search_faiss_memory(staff_id: str, query: str, k: int = 3):
    """Search FAISS vector store for relevant conversation history."""
    try:
        vectorstore = get_faiss_store(staff_id)
        if not vectorstore:
            return []
        
        results = vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]
        
    except Exception as e:
        print(f"❌ FAISS search error for {staff_id}: {e}")
        return []


# ============================================================
# Memory Event Logging
# ============================================================

def log_memory_event(staff_id: str, event_type: str, details: str):
    """Log memory sync events to events.jsonl."""
    try:
        events_file = Path("./events.jsonl")
        lock_file = Path("./events.jsonl.lock")
        
        with FileLock(str(lock_file), timeout=10):
            with open(events_file, "a", encoding="utf-8") as f:
                event = {
                    "at": datetime.utcnow().isoformat() + "Z",
                    "employee": staff_id,
                    "kind": "MEMORY",
                    "event": f"{event_type}: {details}",
                    "type": "memory_sync",
                    "status": "ok"
                }
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠️ Failed to log memory event: {e}")


# ============================================================
# Audio Conversion Helper (FFmpeg)
# ============================================================

def convert_to_mp3(audio_path: str) -> str:
    """
    Convert audio file (ogg, opus, m4a, etc.) to MP3 using FFmpeg.
    WhatsApp often sends ogg/opus which need conversion before Whisper.
    
    Returns:
        Path to converted MP3 file, or original path if conversion fails
    """
    try:
        import subprocess
        
        # Check if already MP3
        if audio_path.lower().endswith('.mp3'):
            return audio_path
        
        # Create output path
        output_path = audio_path.rsplit('.', 1)[0] + '_converted.mp3'
        
        # Run FFmpeg conversion
        cmd = [
            'ffmpeg',
            '-i', audio_path,
            '-acodec', 'libmp3lame',
            '-ar', '16000',  # 16kHz sample rate (good for speech)
            '-ac', '1',       # Mono channel
            '-y',             # Overwrite output file
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30
        )
        
        if result.returncode == 0 and Path(output_path).exists():
            print(f"✅ FFmpeg converted: {audio_path} → {output_path}")
            log_memory_event("system", "audio_conversion", f"FFmpeg: {Path(audio_path).name} → MP3")
            return output_path
        else:
            print(f"⚠️ FFmpeg conversion failed, using original file")
            return audio_path
            
    except Exception as e:
        print(f"⚠️ FFmpeg conversion error: {e}")
        return audio_path  # Fall back to original file


# ============================================================
# Whisper STT Integration (with FFmpeg conversion)
# ============================================================

def whisper_transcribe_audio(audio_path: str, retry: bool = True) -> str:
    """
    Transcribe audio using OpenAI Whisper API.
    Automatically converts ogg/opus to MP3 using FFmpeg before transcription.
    
    Args:
        audio_path: Path to audio file
        retry: Whether to retry once if transcription is empty
    
    Returns:
        Transcribed text or None on failure
    """
    try:
        if not openai_client:
            log_memory_event("system", "stt_error", "OpenAI client not configured")
            return None
        
        # Convert to MP3 if needed (ogg/opus → mp3)
        mp3_path = convert_to_mp3(audio_path)
        
        # Transcribe with Whisper
        with open(mp3_path, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"  # Can be auto-detected or set to specific language
            )
        
        text = transcript.text.strip()
        
        # Retry once if empty transcript
        if not text and retry:
            print("⚠️ Empty transcript, retrying once...")
            return whisper_transcribe_audio(audio_path, retry=False)
        
        if text:
            print(f"[Whisper STT] {text[:100]}...")
            log_memory_event("system", "stt_ok", f"Transcribed {len(text)} chars")
            return text
        else:
            log_memory_event("system", "stt_error", "Empty transcript after retry")
            return None
        
    except Exception as e:
        print(f"❌ Whisper STT error: {e}")
        traceback.print_exc()
        log_memory_event("system", "stt_error", f"Whisper failed: {str(e)}")
        return None


# ============================================================
# Image Analysis with GPT-4o Vision
# ============================================================

def analyze_image_with_gpt(image_path: str, prompt: str = "") -> str:
    """
    Analyzes image using GPT-4o Vision API.
    Returns bilingual description with explicit headers.
    """
    try:
        if not openai_client:
            return None
        
        # Read and encode image as base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Enhanced bilingual prompt with explicit format
        analysis_prompt = prompt or """Analyze this image in detail and provide a summary in the following format:

*Summary (English):*
[Your English analysis here]

*Summary (Roman Urdu):*
[Your Roman Urdu analysis here - use Romanized Urdu script]"""
        
        # Create vision request
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": analysis_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        reply = response.choices[0].message.content
        print(f"[GPT-4o Vision] {reply[:150]}...")
        return reply.strip() if reply else None
        
    except Exception as e:
        print(f"❌ Image analysis error: {e}")
        traceback.print_exc()
        return None


# ============================================================
# LangChain Runnable with Message History and Auto-Retry
# ============================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def _call_ai_with_retry(messages):
    """Call AI with retry logic (up to 3 attempts)."""
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not configured")
    
    response = gpt_model.invoke(messages)
    return response.content.strip()


def get_reply_from_ai(employee_id: str,
                      message_text: str,
                      use_memory: bool = True,
                      mode="reply") -> str:
    """
    GPT-4 handler with LangChain message history and retry logic.
    NOW READS & WRITES: JSONL + PostgreSQL + FAISS (3-layer persistent memory)
    
    Args:
        employee_id: Employee name or ID for memory tracking
        message_text: User's message
        use_memory: Whether to use conversation history
        mode: "reply" or other modes
    
    Returns:
        AI reply text or None on failure
    """
    try:
        # Add bilingual instruction for file/analysis requests
        text_lower = message_text.lower()
        if any(kw in text_lower for kw in ["pdf", "image", "audio", "file", "analyze"]):
            enhanced_prompt = f"{message_text}\n\nPlease provide response in English and Roman Urdu."
            message_text = enhanced_prompt
        
        # Prepare messages
        if use_memory:
            # 🔹 LAYER 1: Get JSONL-based message history (in-session)
            memory = get_employee_memory(employee_id)
            messages = memory.messages.copy()
            
            # 🔹 LAYER 2: Load PostgreSQL context (cross-session persistence)
            postgres_context = load_memory_from_postgres(employee_id)
            
            # 🔹 LAYER 3: Search FAISS for semantically similar past conversations
            faiss_results = search_faiss_memory(employee_id, message_text, k=3)
            
            # Build comprehensive context
            context_parts = []
            
            # Add PostgreSQL historical context if available
            if postgres_context:
                context_parts.append(f"[Previous Conversation Context]:\n{postgres_context}")
            
            # Add FAISS semantic search results if available
            if faiss_results:
                relevant_history = "\n---\n".join(faiss_results[:3])
                context_parts.append(f"[Relevant Past Messages]:\n{relevant_history}")
            
            # Create system message with persistent context
            if context_parts or not messages:
                system_content = "You are a helpful WhatsApp assistant. Provide responses in both English and Roman Urdu when appropriate."
                
                # Append persistent context to system message
                if context_parts:
                    system_content += "\n\n" + "\n\n".join(context_parts)
                
                system_msg = SystemMessage(content=system_content)
                
                # Insert at beginning if no messages, otherwise prepend
                if not messages:
                    messages.append(system_msg)
                elif not any(isinstance(m, SystemMessage) for m in messages):
                    messages.insert(0, system_msg)
            
            # Add current message
            current_msg = HumanMessage(content=message_text)
            messages.append(current_msg)
        else:
            # No memory - just current message
            messages = [
                SystemMessage(content="You are a helpful WhatsApp assistant. Provide responses in both English and Roman Urdu when appropriate."),
                HumanMessage(content=message_text)
            ]
        
        # Call AI with retry logic (now with full persistent context)
        reply = _call_ai_with_retry(messages)
        
        # Save to memory if enabled (3-layer persistent storage)
        if use_memory:
            # 1️⃣ JSONL-based memory (for LangChain in-session)
            memory = get_employee_memory(employee_id)
            memory.add_message(HumanMessage(content=message_text))
            memory.add_message(AIMessage(content=reply))
            
            # 2️⃣ FAISS vector store (for semantic search)
            embed_text_with_faiss(employee_id, message_text, role="human")
            embed_text_with_faiss(employee_id, reply, role="ai")
            
            # 3️⃣ PostgreSQL (for long-term persistence)
            conversation_context = "\n".join([
                f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
                for m in memory.messages[-20:]  # Last 20 messages
            ])
            save_memory_to_postgres(employee_id, conversation_context)
        
        print(f"[AI Reply] {reply[:200]}...")
        return reply
    
    except Exception as e:
        print(f"❌ AI error after retries: {e}")
        traceback.print_exc()
        
        # Return bilingual fallback message
        return "AI temporarily unavailable. Please try again in a moment.\n\nAI abhi kuch der ke liye unavailable hai. Thodi der baad try karein."


# ============================================================
# JSONL → PostgreSQL Migration (Run on Startup)
# ============================================================

def migrate_jsonl_to_postgres():
    """
    Migrate existing JSONL conversation histories to PostgreSQL.
    Runs once on startup to sync historical data.
    """
    try:
        print("\n🔄 Starting JSONL → PostgreSQL migration...")
        
        if not DATABASE_URL:
            print("⚠️ DATABASE_URL not configured, skipping migration")
            return
        
        migrated = 0
        skipped = 0
        
        # Scan all JSONL files in memory directory
        for jsonl_file in MEMORY_DIR.glob("*.jsonl"):
            staff_id = jsonl_file.stem
            
            try:
                # Check if already migrated
                existing_context = load_memory_from_postgres(staff_id)
                if existing_context:
                    skipped += 1
                    continue
                
                # Read JSONL messages
                messages = []
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            record = json.loads(line.strip())
                            role = record.get("role", "human")
                            content = record.get("content", "")
                            messages.append(f"{role.capitalize()}: {content}")
                        except Exception:
                            continue
                
                if messages:
                    # Create context string
                    context = "\n".join(messages[-20:])  # Last 20 messages
                    
                    # Save to PostgreSQL
                    if save_memory_to_postgres(staff_id, context):
                        migrated += 1
                        print(f"  ✅ Migrated {staff_id}: {len(messages)} messages")
                    
            except Exception as e:
                print(f"  ⚠️ Error migrating {staff_id}: {e}")
        
        print(f"✅ Migration complete: {migrated} staff migrated, {skipped} already existed\n")
        
    except Exception as e:
        print(f"❌ Migration error: {e}")
        traceback.print_exc()


# ============================================================
# Initialization: Run migration on import
# ============================================================

# Auto-run migration when module is imported
if DATABASE_URL and embeddings_model:
    migrate_jsonl_to_postgres()
    print("✅ FAISS + PostgreSQL Persistent Memory Ready — staff context now stored permanently.")
