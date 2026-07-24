# ⚡ DataDealers — Professional Data Science Platform

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.org/)
[![scikit-learn](https://img.shields.io/badge/scikit_learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Render](https://img.shields.io/badge/Render-Backend-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com)
[![Cloudflare Pages](https://img.shields.io/badge/Cloudflare_Pages-Frontend-F38020?style=for-the-badge&logo=cloudflare&logoColor=white)](https://pages.cloudflare.com/)

DataDealers is a full-stack data science platform designed for automated dataset cleaning, quality auditing, machine learning model evaluation, and AI-driven model guidance. 

It pairs an interactive web frontend with a powerful Python backend capable of running real machine learning workloads using **scikit-learn**, **pandas**, and **XGBoost**, supplemented by **Google Gemini API** for intelligent problem formulation.

---

## 🌟 Key Features

* 🔐 **Minimal Auth Gate**: Lightweight user registration and login with local database storage (`SQLite`).
* 📝 **Problem Formulation**: Interactive natural language input for data science objectives.
* 🤖 **AI Model Guidance**: Integrates server-side with **Google Gemini 1.5 Flash** to recommend optimal model families, input formats, and plain-English explanations with adjustable simplicity levels.
* 🧹 **Automated Data Cleaning**: Interactive type coercion, missing value imputation (mean/median/mode), duplicate removal, and Z-score / IQR outlier filtering.
* 🛡️ **Automated Quality Checks**: 7-point dataset health inspection (duplicates, zero-variance columns, extreme outliers, target integrity, class balance ratio).
* 🧪 **Model Testing & Training**: Real machine learning execution (Random Forest, XGBoost, Logistic Regression, SVM, Decision Trees) with live accuracy, precision, recall, F1, and cross-validation logs.
* 📦 **Dataset Supply**: Download and explore curated sample datasets (Iris, Wine, Breast Cancer, Diabetes).
* 🌓 **Adaptive Design System**: Dark/Light mode theme toggle built with vanilla CSS tokens and sleek typography (Inter, IBM Plex Mono, Syne).

---

## 🏗️ Tech Stack & Architecture

### **Frontend**
* **Core**: Vanilla HTML5, CSS3, JavaScript (ES6+ Fetch API).
* **Typography**: Inter, Inter Tight, IBM Plex Mono, Syne.
* **Architecture**: Single-Page App (SPA) layout with responsive sidebar navigation and inline modal overlays.

### **Backend**
* **Framework**: Python 3.11 + Flask, Flask-CORS, Gunicorn.
* **Data & ML**: `pandas`, `numpy`, `scikit-learn`, `xgboost`.
* **Database**: `SQLite` (via lightweight `db.py` wrapper) + `quotes_seed.py`.
* **Generative AI**: Google Gemini API via REST requests.

---

## 📁 Repository Structure

```text
datadealers/
├── index.html           # Main production frontend SPA
├── app.py               # Flask REST API backend
├── db.py                # SQLite database management
├── quotes_seed.py       # Seed data for inspiring quotes
├── requirements.txt     # Python dependencies
├── render.yaml          # Render deployment manifest
├── Procfile.txt         # Process file for Gunicorn web server
├── runtime.txt          # Python runtime version definition
├── data-library/        # Directory for sample datasets
└── dist/                # Production build distribution directory
    └── index.html       # Static production frontend bundle
```

---

## 🚀 Quickstart (Local Development)

### 1. Backend Setup
```bash
# Clone the repository
git clone https://github.com/adityachourasia10329/datadealers.git
cd datadealers

# Install dependencies
pip install -r requirements.txt

# (Optional) Set your Gemini API Key in .env file or environment
echo "GEMINI_API_KEY=your_actual_api_key_here" > .env

# Run the Flask development server
python app.py
```
The backend server will start on `http://localhost:5000`.

### 2. Frontend Setup
Simply open `index.html` directly in your browser, or serve it locally using Python:
```bash
python -m http.server 8000
```
Open `http://localhost:8000` in your web browser.

---

## 🌐 Production Deployment

### Backend Deployment (Render)
The repository includes a ready-to-use `render.yaml` configuration:
1. Connect your repository to **[Render](https://render.com)**.
2. Render will automatically detect `render.yaml` and configure:
   * **Build Command**: `pip install -r requirements.txt`
   * **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
3. Add environment variable `GEMINI_API_KEY` under the **Environment** settings tab in Render.

### Frontend Deployment (Cloudflare Pages / Vercel / Netlify)
Deploy the frontend repository to any static hosting provider (e.g., **Cloudflare Pages**):
1. Connect your GitHub repository to Cloudflare Pages.
2. Set **Build Output Directory** to `/`.
3. Click **Save and Deploy**.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Health check & service status |
| `GET` | `/quote` | Returns a random inspiring quote |
| `POST` | `/signup` | User account creation |
| `POST` | `/login` | User authentication |
| `POST` | `/upload_data` | Upload and analyze dataset structure |
| `POST` | `/upload_model` | Upload custom model artifact (.pkl, .py, .js) |
| `POST` | `/clean/<sid>` | Apply dataset cleaning and outlier filters |
| `GET` | `/quality_tests/<sid>` | Execute 7 quality assurance audits |
| `POST` | `/split/<sid>` | Train/test dataset split with stratification |
| `POST` | `/guidance` | Server-side Gemini AI model recommendation |
| `POST` | `/test_model` | Train and evaluate scikit-learn / XGBoost model |

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
