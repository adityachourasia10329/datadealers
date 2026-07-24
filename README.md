# Data Dealers — Backend

Flask + pandas + scikit-learn API that powers the Data Dealers frontend
(`data-dealers-app.html`). All endpoints do real work: real CSV parsing,
real cleaning, real quality checks, real train/test splits, and real
model training with scikit-learn / XGBoost — nothing here is faked.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

The API starts on `http://localhost:5000`. Open `data-dealers-app.html`
in a browser — it talks to `localhost:5000` by default, so it should
just work.

## Deploy to Render (recommended, free tier available)

1. Push this `backend/` folder to a GitHub repo (or a `backend/` subfolder
   of your existing repo).
2. Go to [render.com](https://render.com) → **New +** → **Web Service** →
   connect your repo.
3. Render will auto-detect `render.yaml` in this folder. If it doesn't,
   set manually:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
4. Deploy. Render gives you a URL like `https://data-dealers-api.onrender.com`.
5. Open `data-dealers-app.html`, tap the connection pill (top-right,
   says "Demo" or "Live"), and paste in your Render URL. It's saved in
   the browser so you only need to do this once per device.

Note: Render's free tier spins down after inactivity, so the first
request after idling can take ~30-60s to wake up.

## Deploy to Railway

1. [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**.
2. Point it at this `backend/` folder.
3. Railway auto-detects the `Procfile` and installs `requirements.txt`.
4. Once deployed, copy the public URL Railway gives you and paste it
   into the app's connection pill, same as above.

## Notes on storage (important for production use)

- **Users** (`USERS` dict) and **uploaded datasets/sessions** (`SESSIONS`
  dict) are stored **in memory**. That means:
  - Restarting the server (or a free-tier host spinning down/up) wipes
    all accounts and uploaded sessions.
  - This is fine for demos, coursework, or a personal tool, but **not**
    fine for real users' data. If you want persistence, swap:
    - `USERS` → a real table (Postgres/SQLite) with proper password
      hashing (already using `werkzeug.security`, just needs a DB).
    - `SESSIONS` → store the DataFrame as a file (e.g. `parquet`) on
      disk or in S3, keyed by session id, instead of keeping it in
      process memory.
- CORS is wide open (`CORS(app)`) so the static frontend can call it
  from anywhere. Lock this down to your actual frontend's origin
  before using this with real user data.

## Endpoints

| Method | Path                         | Purpose                                   |
|--------|------------------------------|--------------------------------------------|
| GET    | `/`                          | Health check                               |
| POST   | `/signup`                    | Create account                             |
| POST   | `/login`                     | Log in                                     |
| POST   | `/upload_data`               | Upload + analyze a CSV                     |
| POST   | `/clean/<session_id>`        | Apply cleaning options                     |
| GET    | `/quality_tests/<session_id>`| Run 7 quality checks                       |
| POST   | `/split/<session_id>`        | Train/test split                           |
| POST   | `/guidance`                  | Rule-based model recommendations           |
| POST   | `/test_model`                | Train & evaluate a real scikit-learn model |
| GET    | `/datasets`                  | List curated sample datasets               |
| GET    | `/datasets/<name>/download`  | Download a sample dataset as CSV           |
