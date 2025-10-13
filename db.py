"""Async SQLAlchemy PostgreSQL helper for the project.

- Uses DATABASE_URL from env (asyncpg) or falls back to postgresql+asyncpg://postgres:postgres@localhost:5432/what_agent
- Provides AsyncEngine, AsyncSession, Base
- Two simple models: Employee, Task mirroring existing JSON structures
- init_db(engine) creates tables and migrates employees.json into the database if empty
"""
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker, Mapped, mapped_column
from sqlalchemy import Integer, String, Text, Boolean, DateTime, select
from sqlalchemy.sql import func

DATABASE_URL = os.getenv("DATABASE_URL") or "postgresql+asyncpg://postgres:postgres@localhost:5432/what_agent"

Base = declarative_base()

engine: Optional[AsyncEngine] = None
SessionLocal: Optional[sessionmaker] = None

class Employee(Base):
    __tablename__ = "employees"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    msisdn: Mapped[str] = mapped_column(String(32), nullable=False)
    pref: Mapped[str] = mapped_column(String(16), nullable=False, server_default="auto")
    meta: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), server_default=func.now())

class Task(Base):
    __tablename__ = "tasks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    employee_id: Mapped[int] = mapped_column(Integer, nullable=True)
    title: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, server_default="pending")
    created_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), server_default=func.now())

async def get_engine() -> AsyncEngine:
    global engine
    if engine is None:
        # Try primary DATABASE_URL. If connection fails at init time, init_db() will
        # replace the engine with a local SQLite async engine as a fallback so the
        # app can run without a Postgres server during development/testing.
        engine = create_async_engine(DATABASE_URL, echo=False, future=True)
    return engine

async def get_session() -> AsyncSession:
    global SessionLocal
    if SessionLocal is None:
        eng = await get_engine()
        SessionLocal = sessionmaker(bind=eng, class_=AsyncSession, expire_on_commit=False)
    return SessionLocal()

async def init_db(migrate_employees_from: Path | None = Path("./employees.json")) -> None:
    """Create tables and optionally migrate employees.json into the employees table if table is empty."""
    eng = await get_engine()
    # Try creating tables on the configured engine. If that fails (e.g. Postgres
    # not reachable), fall back to a local SQLite async engine and continue.
    try:
        async with eng.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception:
        # Fallback to sqlite+aiosqlite for development if primary DB is not reachable
        sqlite_url = "sqlite+aiosqlite:///./what_agent.db"
        global engine
        engine = create_async_engine(sqlite_url, echo=False, future=True)
        # replace global sessionmaker
        global SessionLocal
        SessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    # Migrate employees.json
    try:
        if migrate_employees_from and migrate_employees_from.exists():
            with open(migrate_employees_from, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {}
    except Exception:
        data = {}

    # Insert into DB if table is empty
    async with await get_session() as session:
        # Use SQLAlchemy select() to be compatible with SQLAlchemy 2.0 execution rules
        res = await session.execute(select(func.count()).select_from(Employee))
        count = res.scalar_one_or_none() or 0
        if count == 0 and data:
            for name, val in data.items():
                if isinstance(val, str):
                    msisdn = val
                    pref = os.getenv("DELIVERY_DEFAULT", "auto")
                else:
                    msisdn = val.get("msisdn")
                    pref = val.get("pref", os.getenv("DELIVERY_DEFAULT", "auto"))
                emp = Employee(name=name, msisdn=msisdn, pref=pref)
                session.add(emp)
            await session.commit()


async def close_engine():
    global engine
    if engine is not None:
        await engine.dispose()
        engine = None
