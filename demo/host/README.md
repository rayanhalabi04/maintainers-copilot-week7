# Host Demo

Static host page that embeds the Maintainer's Copilot widget through the backend loader script.

## Run backend

```bash
PYTHONPATH=backend/model_server uv run uvicorn app.main:app --reload --port 8001
```

## Run widget dev server

```bash
cd frontend/widget
npm install
npm run dev
```

## Open host page

From the repo root, open `demo/host/index.html` in a browser, or serve it with a tiny static server:

```bash
python3 -m http.server 8080 --directory demo/host
```

Then visit `http://localhost:8080`.

## Demo credentials

- `user@example.com` / `user123`
- `admin@example.com` / `admin123`
