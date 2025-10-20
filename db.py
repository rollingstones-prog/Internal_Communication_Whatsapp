"""
Async SQLAlchemy PostgreSQL helper for the project
— adds WhatsAppInbox model and message helpers.
"""
from __future__ import annotations
import os, json
from pathlib import Path
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker, Mapped, mapped_column
from sqlalchemy import Integer, String, Text, Boolean, DateTime, select, update
from sqlalchemy.sql import func
from dotenv import load_dotenv # <-- Zaroori Import

# Load environment variables (Render par yeh zyada zaroori nahi, lekin local testing ke liye achha hai)
load_dotenv(override=True)

# 🛑 CRITICAL FIX: DATABASE_URL ko Environment Variable se load karein
# Agar DATABASE_URL set hai (PostgreSQL), toh woh use hoga.
# Agar set nahi hai, toh yeh SQLite par fallback karega.
# NOTE: Ab yeh line PostgreSQL URL (jo aapne set kiya hai) use karegi.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./what_agent.db")

Base = declarative_base()
engine: Optional[AsyncEngine] = None
SessionLocal: Optional[sessionmaker] = None


# ───────────────────────────────────────────
# MODELS
# ───────────────────────────────────────────
class Employee(Base):
    __tablename__ = "employees"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    msisdn: Mapped[str] = mapped_column(String(32), nullable=False)
    pref: Mapped[str] = mapped_column(String(16),
                                     nullable=False,
                                     server_default="auto")
    meta: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[Optional[DateTime]] = mapped_column(
        DateTime(timezone=True), server_default=func.now())


class Task(Base):
    __tablename__ = "tasks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    employee_id: Mapped[int] = mapped_column(Integer, nullable=True)
    title: Mapped[Optional[str]] = mapped_column(String(256))
    details: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32),
                                     nullable=False,
                                     server_default="pending")
    created_at: Mapped[Optional[DateTime]] = mapped_column(
        DateTime(timezone=True), server_default=func.now())


class WhatsAppInbox(Base):
    __tablename__ = "whatsapp_inbox"
    id: Mapped[int] = mapped_column(Integer,
                                     primary_key=True,
                                     autoincrement=True)
    phone: Mapped[str] = mapped_column(String(32), nullable=False)
    message_text: Mapped[str] = mapped_column(Text, nullable=False)
    processed: Mapped[bool] = mapped_column(Boolean,
                                         nullable=False,
                                         server_default="false")
    created_at: Mapped[Optional[DateTime]] = mapped_column(
        DateTime(timezone=True), server_default=func.now())


# ───────────────────────────────────────────
# CONNECTION + HELPERS (Yeh Hissa Wahi Rahega)
# ───────────────────────────────────────────
async def get_engine() -> AsyncEngine:
    global engine
    if engine is None:
        # Ab yeh line sahi DATABASE_URL (Postgres) use karegi
        engine = create_async_engine(DATABASE_URL, echo=False, future=True)
    return engine


async def get_session() -> AsyncSession:
    global SessionLocal
    if SessionLocal is None:
        eng = await get_engine()
        SessionLocal = sessionmaker(bind=eng,
                                     class_=AsyncSession,
                                     expire_on_commit=False)
    return SessionLocal()


async def init_db(migrate_employees_from: Path
                 | None = Path("./employees.json")) -> None:
    """Create tables and optionally migrate employees.json."""
    eng = await get_engine()
    try:
        # Postgres connection try karein
        async with eng.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:
        print(f"PostgreSQL connection/init failed. Falling back to SQLite: {e}")
        # Agar Postgres unavailable, toh SQLite par fallback karein
        sqlite_url = "sqlite+aiosqlite:///./what_agent.db"
        global engine
        engine = create_async_engine(sqlite_url, echo=False, future=True)
        global SessionLocal
        SessionLocal = sessionmaker(bind=engine,
                                     class_=AsyncSession,
                                     expire_on_commit=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    # Optional migration of employees.json
    try:
        data = json.load(open(migrate_employees_from)
                             ) if migrate_employees_from.exists() else {}
    except Exception:
        data = {}

    async with await get_session() as session:
        res = await session.execute(select(func.count()).select_from(Employee))
        count = res.scalar_one_or_none() or 0
        if count == 0 and data:
            for name, val in data.items():
                msisdn = val["msisdn"] if isinstance(val, dict) else val
                pref = val.get("pref", os.getenv("DELIVERY_DEFAULT",
                                                     "auto")) if isinstance(
                                                     val, dict) else "auto"
                session.add(Employee(name=name, msisdn=msisdn, pref=pref))
            await session.commit()


async def get_new_messages():
    """Return list of unprocessed WhatsApp messages."""
    async with await get_session() as session:
        result = await session.execute(
            select(WhatsAppInbox).where(WhatsAppInbox.processed == False))
        rows = result.scalars().all()
        return [{
            "inbox_id": r.id,
            "phone": r.phone,
            "message_text": r.message_text,
            "at": r.created_at # Add created_at for dashboard_api
        } for r in rows]


async def mark_message_processed(inbox_id: int):
    """Mark message as processed."""
    async with await get_session() as session:
        await session.execute(
            update(WhatsAppInbox).where(WhatsAppInbox.id == inbox_id).values(
                processed=True))
        await session.commit()


async def close_engine():
    global engine
    if engine is not None:
        await engine.dispose()
        engine = None
