# Maintainer's Copilot Widget

Minimal Vite + React embeddable widget for the Week 7 Maintainer's Copilot.

## Start the backend

```bash
PYTHONPATH=backend/model_server uv run uvicorn app.main:app --reload --port 8001
```

The backend serves:

- `GET /widget.js`
- `GET /widget/config/demo-widget`
- `POST /auth/login`
- `POST /chat`

## Start the widget dev server

```bash
cd frontend/widget
npm install
npm run dev
```

The widget dev server runs on `http://localhost:5173`.

## Demo credentials

- `user@example.com` / `user123`
- `admin@example.com` / `admin123`

## Notes

This is a minimal local/dev widget. It supports runtime config, login, chat, tool-call display, and iframe resize via `postMessage`. It is not a production widget config database, tenant allowlist, analytics, Vault, or MinIO implementation.
