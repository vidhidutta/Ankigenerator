# OjaMed API (FastAPI)

Minimal FastAPI service for OjaMed "upload -> generate deck" flow.
- `POST /convert` with `file` (PPT/PPTX/PDF) returns a ZIP containing:
  - `deck.apkg` and `deck.csv` (CSV is a reliable fallback).
- `GET /health` returns `{"ok": true}`.

## Local dev

```bash
# one-time
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

# run
uvicorn app.main:app --reload --port 8000
# open http://127.0.0.1:8000/docs
```

Test:

```bash
pytest -q
```

## Environment

* `MAX_FILE_MB` (default 50)
* `ALLOWED_ORIGINS` comma-separated (default "\*"). For prod, set:
  `https://app.ojamed.com,https://ojamed.com,https://<your>.pages.dev`

## Deploy to Render (Free)

1. Push this repo to GitHub.
2. In Render: **New > Web Service > Build & Deploy from GitHub**.
3. Runtime: Python 3.11
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
6. Env Vars:

   * `MAX_FILE_MB=50`
   * `ALLOWED_ORIGINS=https://app.ojamed.com,https://ojamed.com,https://<your>.pages.dev`
7. Deploy. Your API URL will look like:
   `https://<service-name>.onrender.com`

## Frontend usage (example fetch)

```js
async function convert(file) {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch("https://<service>.onrender.com/convert", {
    method: "POST",
    body: fd
  });
  if (!r.ok) throw new Error(await r.text());
  const blob = await r.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "ojamed_deck.zip"; a.click();
  URL.revokeObjectURL(url);
}
```

## Swap in your real generator

Edit `app/pipeline.py` → implement `run_pipeline(input_path) -> (apkg_path, csv_path)`.
# Force deployment Tue Sep  2 12:02:41 PM BST 2025
