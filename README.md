What-Agent / Internal Communication WhatsApp
=========================================

Postgres integration
--------------------

This project can use a Postgres database to store employees and tasks. It uses async SQLAlchemy (SQLAlchemy 2.0) with asyncpg.

1. Create a Postgres database locally or remotely. Example:

	- database: what_agent
	- user: postgres
	- password: postgres

2. Set the DATABASE_URL environment variable (example):

	DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/what_agent

3. Install dependencies (recommended to use a virtualenv):

	pip install -r requirements.txt

4. Start the app (uvicorn). The app will run DB initialization on startup and migrate `employees.json` into the `employees` table if empty.

	uvicorn main:app --host 0.0.0.0 --port 8000

Notes
-----
- If you already have an `employees.json` file, the first startup will insert those rows into the database if the `employees` table is empty.
- If you prefer migrations, consider adding Alembic and migrating schemas separately.
