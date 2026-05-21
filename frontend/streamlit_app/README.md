# Maintainer's Copilot Streamlit App

Minimal internal Streamlit app for the Week 7 Maintainer's Copilot backend.
Chat, memory, and admin views require logging in with a demo account.

## Start the backend

```bash
PYTHONPATH=backend/model_server uv run uvicorn app.main:app --reload --port 8001
```

## Start Streamlit

```bash
MODEL_SERVER_URL=http://localhost:8001 uv run streamlit run frontend/streamlit_app/app.py
```

`MODEL_SERVER_URL` defaults to `http://localhost:8001`.

## Demo logins

- `admin@example.com` / `admin123`
- `user@example.com` / `user123`

Auth and memory are local/demo implementations. This app is an internal tool only; it is not a production auth, Vault, React widget, or hosted widget implementation.
