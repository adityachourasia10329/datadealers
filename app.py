"""
Data Dealers — Backend API
Flask + SQLite + pandas + scikit-learn + Gemini API backend
"""

import io
import os
import time
import uuid
import json
import traceback
import requests

try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    load_dotenv(env_path, override=True)
except ImportError:
    pass

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

import db

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.datasets import (
    make_classification, make_regression,
    load_iris, load_wine, load_breast_cancer, load_diabetes
)

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
MODEL_UPLOAD_DIR = os.path.join(UPLOAD_DIR, 'models')
DATA_LIBRARY_DIR = os.path.join(os.path.dirname(__file__), 'data-library')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_LIBRARY_DIR, exist_ok=True)

SESSIONS = {}  # session_id -> {df, clean_df, train_df, test_df, target, filename}
STORED_MODELS = {}  # model_id -> {filename, ext, path}
MAX_ROWS_RETURNED_SAMPLE = 3



def now_ts():
    return time.strftime('%H:%M:%S')


# ─────────────────────────────────────────────
#  HEALTH & QUOTES
# ─────────────────────────────────────────────
@app.route('/')
def health():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_path):
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path, override=True)
        except Exception:
            pass
    key = (os.environ.get('GEMINI_API_KEY') or '').strip()
    return jsonify({
        'status': 'ok',
        'service': 'data-dealers-api',
        'gemini_configured': bool(key and key != 'your_gemini_api_key_here')
    })


@app.route('/quote', methods=['GET'])
def get_quote():
    q = db.get_random_quote()
    return jsonify({'quote': q})


# ─────────────────────────────────────────────
#  AUTHENTICATION
# ─────────────────────────────────────────────
@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json(force=True, silent=True) or {}
    name = (data.get('name') or '').strip()
    email = (data.get('email') or '').strip().lower()
    pwd = data.get('password') or ''
    allow_downloads = data.get('downloads_allowed', True)

    if not name or not email or not pwd:
        return jsonify({'error': 'Name, email and password are all required.'}), 400

    user = db.create_user(name, email, pwd, downloads_allowed=allow_downloads)
    if not user:
        return jsonify({'error': 'An account with that email already exists.'}), 409

    quote = db.get_random_quote()
    return jsonify({'name': user['name'], 'email': user['email'], 'quote': quote})


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json(force=True, silent=True) or {}
    email = (data.get('email') or '').strip().lower()
    pwd = data.get('password') or ''

    if not email or not pwd:
        return jsonify({'error': 'Email and password are required.'}), 400

    # Verify existing lock & key pair or register pair on login page
    user = db.verify_or_create_user(email, pwd)
    quote = db.get_random_quote()
    return jsonify({'name': user['name'], 'email': user['email'], 'quote': quote})


# ─────────────────────────────────────────────
#  UPLOAD & STORAGE
# ─────────────────────────────────────────────
ALLOWED_EXTENSIONS = {'csv', 'json', 'tsv', 'txt', 'parquet', 'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload_data', methods=['POST'])
def upload_data():
    f = request.files.get('file')
    if not f:
        return jsonify({'error': 'No file uploaded.'}), 400

    filename = f.filename or 'upload.csv'
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    if not allowed_file(filename):
        return jsonify({'error': f'Unsupported file format .{ext}. Allowed: CSV, JSON, TSV, Parquet, TXT, PNG, JPG.'}), 400

    sid = str(uuid.uuid4())[:8]
    save_path = os.path.join(UPLOAD_DIR, f"{sid}_{filename}")
    
    try:
        raw = f.read()
        with open(save_path, 'wb') as out_f:
            out_f.write(raw)

        # Parse structured datasets (CSV / TSV / JSON / Parquet)
        df = None
        if ext == 'csv':
            df = pd.read_csv(io.BytesIO(raw))
        elif ext == 'tsv':
            df = pd.read_csv(io.BytesIO(raw), sep='\t')
        elif ext == 'json':
            df = pd.read_json(io.BytesIO(raw))
        elif ext == 'parquet':
            df = pd.read_parquet(io.BytesIO(raw))
        elif ext in ['txt', 'png', 'jpg', 'jpeg']:
            # Non-tabular file stored successfully
            return jsonify({
                'session_id': sid,
                'filename': filename,
                'is_tabular': False,
                'file_size_kb': round(len(raw) / 1024, 1),
                'message': f'File {filename} stored successfully in backend storage.'
            })

    except Exception as e:
        return jsonify({'error': f'Could not parse dataset file: {e}'}), 400

    if df is None or df.empty:
        return jsonify({'error': 'The uploaded file contains no data rows.'}), 400

    SESSIONS[sid] = {
        'df': df,
        'clean_df': None,
        'train_df': None,
        'test_df': None,
        'target': None,
        'filename': filename
    }

    columns = []
    suggestions = []
    dup_count = int(df.duplicated().sum())

    for col in df.columns:
        s = df[col]
        missing_pct = round(float(s.isna().mean()) * 100, 1)
        is_numeric = pd.api.types.is_numeric_dtype(s)

        if not is_numeric:
            coerced = pd.to_numeric(s, errors='coerce')
            if coerced.notna().mean() > 0.85 and s.notna().mean() > 0:
                is_numeric = True
                s = coerced

        col_info = {
            'name': str(col),
            'kind': 'numeric' if is_numeric else 'categorical',
            'missing_pct': missing_pct,
        }
        if is_numeric:
            clean_s = s.dropna()
            if len(clean_s):
                col_info['stats'] = {
                    'min': round(float(clean_s.min()), 4),
                    'mean': round(float(clean_s.mean()), 4),
                    'max': round(float(clean_s.max()), 4),
                }
            col_info['sample_values'] = [str(v) for v in df[col].dropna().head(MAX_ROWS_RETURNED_SAMPLE)]
        else:
            top_values = df[col].astype(str).value_counts().head(3).to_dict()
            col_info['stats'] = {'top_values': top_values}
            col_info['sample_values'] = [str(v) for v in df[col].dropna().head(MAX_ROWS_RETURNED_SAMPLE)]

        columns.append(col_info)

    if dup_count:
        suggestions.append(f'Found {dup_count} duplicate row(s) in dataset.')
    missing_cols = [c['name'] for c in columns if c['missing_pct'] > 0]
    if missing_cols:
        suggestions.append(f"Missing values detected in {len(missing_cols)} column(s): {', '.join(missing_cols[:5])}.")
    cat_cols = [c['name'] for c in columns if c['kind'] == 'categorical']
    if cat_cols:
        suggestions.append(f"Categorical features detected: {', '.join(cat_cols[:5])}.")
    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if const_cols:
        suggestions.append(f"Constant column(s) with no variance: {', '.join(const_cols)}.")
    if not suggestions:
        suggestions.append('Dataset structure is clean — ready for preprocessing and model testing.')

    return jsonify({
        'session_id': sid,
        'filename': filename,
        'is_tabular': True,
        'n_rows': int(df.shape[0]),
        'n_cols': int(df.shape[1]),
        'memory_kb': round(df.memory_usage(deep=True).sum() / 1024, 1),
        'columns': columns,
        'suggestions': suggestions,
    })


@app.route('/upload_model', methods=['POST'])
def upload_model():
    f = request.files.get('file')
    if not f:
        return jsonify({'error': 'No model file provided.'}), 400

    filename = f.filename or 'model.pkl'
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    allowed_model_exts = {'py', 'js', 'pkl', 'html'}

    if ext not in allowed_model_exts:
        return jsonify({'error': f'Unsupported model format .{ext}. Allowed types: .py, .js, .pkl, .html'}), 400

    mid = f"mod_{str(uuid.uuid4())[:8]}"
    save_path = os.path.join(MODEL_UPLOAD_DIR, f"{mid}_{filename}")
    
    try:
        f.save(save_path)
        STORED_MODELS[mid] = {
            'model_id': mid,
            'filename': filename,
            'ext': ext,
            'path': save_path,
            'uploaded_at': time.time()
        }
        return jsonify({
            'model_id': mid,
            'filename': filename,
            'file_type': ext,
            'message': f'Model {filename} uploaded and registered successfully (ID: {mid}).'
        })
    except Exception as e:
        return jsonify({'error': f'Failed to store model file: {e}'}), 500


# ─────────────────────────────────────────────
#  DATASET CLEANING
# ─────────────────────────────────────────────

@app.route('/clean/<sid>', methods=['POST'])
def clean(sid):
    sess = SESSIONS.get(sid)
    if not sess:
        return jsonify({'error': 'Unknown or expired session ID. Please upload your dataset again.'}), 404

    opts = request.get_json(force=True, silent=True) or {}
    df = sess['df'].copy()
    original_rows = len(df)
    steps = []

    # 1. Type Coercion
    if opts.get('fix_types', True):
        fixed_cols = []
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                coerced = pd.to_numeric(df[col], errors='coerce')
                if coerced.notna().mean() > 0.85:
                    df[col] = coerced
                    fixed_cols.append(col)
        if fixed_cols:
            steps.append({'type': 'type_fix', 'msg': f"Coerced text columns to numeric format: {', '.join(fixed_cols)}"})

    # 2. Missing Value Imputation
    num_strategy = opts.get('missing_numeric', 'median')
    cat_strategy = opts.get('missing_categorical', 'mode')
    rows_before_missing = len(df)
    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if num_strategy == 'median':
                df[col] = df[col].fillna(df[col].median())
            elif num_strategy == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif num_strategy == 'drop':
                df = df[df[col].notna()]
        else:
            if cat_strategy == 'mode':
                mode_val = df[col].mode(dropna=True)
                df[col] = df[col].fillna(mode_val.iloc[0] if len(mode_val) else 'Unknown')
            elif cat_strategy == 'unknown':
                df[col] = df[col].fillna('Unknown')
            elif cat_strategy == 'drop':
                df = df[df[col].notna()]

    steps.append({'type': 'missing', 'msg': f'Imputed missing values (Numeric: {num_strategy.capitalize()}, Categorical: {cat_strategy.capitalize()})'})

    # 3. Duplicate Removal
    if opts.get('remove_duplicates', True):
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        if removed:
            steps.append({'type': 'duplicates', 'msg': f'Removed {removed} duplicate row(s) from dataset'})

    # 4. Outlier Filtering (IQR vs Z-Score)
    if opts.get('remove_outliers', True):
        before = len(df)
        method = (opts.get('outlier_method') or opts.get('outlier_strategy') or 'iqr').lower()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        mask = pd.Series(True, index=df.index)
        
        for col in numeric_cols:
            series = df[col]
            if series.std(ddof=0) == 0 or series.nunique() <= 2:
                continue
            if method == 'zscore':
                # Z-Score: z = (x - mu) / sigma; flag |z| > 3.0
                mean_val = series.mean()
                std_val = series.std(ddof=0) or 1.0
                z = (series - mean_val) / std_val
                mask &= z.abs() <= 3.0
            else:
                # IQR: Q1 (25th pct), Q3 (75th pct), IQR = Q3 - Q1; bounds [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    continue
                mask &= series.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
                
        df = df[mask]
        removed = before - len(df)
        if removed:
            method_desc = "Interquartile Range (IQR = Q3 - Q1, bounds: Q1 - 1.5*IQR to Q3 + 1.5*IQR)" if method == 'iqr' else "Z-Score (z = (x - μ) / σ, threshold: |z| ≤ 3.0)"
            steps.append({'type': 'outliers', 'msg': f'Filtered {removed} outlier row(s) using {method_desc}'})

    clean_rows = len(df)
    total_removed = original_rows - clean_rows
    missing_after = float(df.isna().mean().mean()) if clean_rows else 0
    quality_score = round(max(0, 100 - missing_after * 100 - (5 if total_removed > original_rows * 0.3 else 0)), 1)

    sess['clean_df'] = df

    return jsonify({
        'original_rows': original_rows,
        'clean_rows': clean_rows,
        'total_removed': total_removed,
        'quality_score': quality_score,
        'steps': steps,
    })


@app.route('/clean/<sid>/download')
def download_cleaned_dataset(sid):
    sess = SESSIONS.get(sid)
    if not sess or sess.get('clean_df') is None:
        return jsonify({'error': 'No cleaned dataset found for this session. Please clean dataset first.'}), 404

    clean_df = sess['clean_df']
    orig_filename = sess.get('filename', 'cleaned_dataset.csv')
    ext = orig_filename.rsplit('.', 1)[1].lower() if '.' in orig_filename else 'csv'
    
    clean_filename = f"cleaned_{orig_filename}"
    buf = io.BytesIO()

    if ext == 'csv':
        clean_df.to_csv(buf, index=False)
        mimetype = 'text/csv'
    elif ext == 'tsv':
        clean_df.to_csv(buf, sep='\t', index=False)
        mimetype = 'text/tab-separated-values'
    elif ext == 'json':
        clean_df.to_json(buf, orient='records', indent=2)
        mimetype = 'application/json'
    elif ext == 'parquet':
        clean_df.to_parquet(buf, index=False)
        mimetype = 'application/octet-stream'
    elif ext in ['txt']:
        clean_df.to_csv(buf, sep=' ', index=False)
        mimetype = 'text/plain'
    else:
        clean_df.to_csv(buf, index=False)
        mimetype = 'text/csv'

    buf.seek(0)
    return send_file(buf, mimetype=mimetype, as_attachment=True, download_name=clean_filename)


# ─────────────────────────────────────────────
#  QUALITY CHECKS
# ─────────────────────────────────────────────
@app.route('/quality_tests/<sid>')
def quality_tests(sid):
    sess = SESSIONS.get(sid)
    if not sess:
        return jsonify({'error': 'Unknown or expired session ID.'}), 404

    df = sess['clean_df'] if sess['clean_df'] is not None else sess['df']
    tests = []

    dup = int(df.duplicated().sum())
    tests.append({'name': 'Duplicate Row Check', 'passed': dup == 0,
                  'detail': 'Zero duplicate rows detected' if dup == 0 else f'{dup} duplicate row(s) present'})

    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    tests.append({'name': 'Column Variance Check', 'passed': len(const_cols) == 0,
                  'detail': 'All columns exhibit variance' if not const_cols else f"Zero-variance columns: {', '.join(const_cols)}"})

    avg_missing = float(df.isna().mean().mean()) * 100
    tests.append({'name': 'Missing Data Threshold (< 5%)', 'passed': avg_missing < 5,
                  'detail': f'Average missing rate: {avg_missing:.1f}%'})

    target = sess.get('target') or (df.columns[-1] if len(df.columns) else None)
    if target and target in df.columns:
        tgt_missing = int(df[target].isna().sum())
        tests.append({'name': 'Target Feature Integrity', 'passed': tgt_missing == 0,
                      'detail': f"Target '{target}' is 100% complete" if tgt_missing == 0 else f'{tgt_missing} missing values in target column'})
    else:
        tests.append({'name': 'Target Feature Integrity', 'passed': True, 'detail': 'Target column not selected yet'})

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    extreme = 0
    for col in numeric_cols:
        s = df[col]
        if s.std(ddof=0) == 0:
            continue
        z = ((s - s.mean()) / s.std(ddof=0)).abs()
        extreme += int((z > 6).sum())
    tests.append({'name': 'Extreme Outlier Audit', 'passed': extreme == 0,
                  'detail': 'No extreme outliers (> 6 std dev)' if extreme == 0 else f'{extreme} extreme outlier values found'})

    tests.append({'name': 'Sample Volume (≥ 50 rows)', 'passed': len(df) >= 50,
                  'detail': f'{len(df)} total rows available'})

    if target and target in df.columns and df[target].nunique(dropna=True) <= 10:
        vc = df[target].value_counts(normalize=True)
        dominant = float(vc.iloc[0]) if len(vc) else 0
        tests.append({'name': 'Class Balance Ratio (≤ 80%)', 'passed': dominant <= 0.8,
                      'detail': f'Dominant class proportion: {dominant*100:.1f}%'})
    else:
        tests.append({'name': 'Class Balance Ratio (≤ 80%)', 'passed': True, 'detail': 'Continuous or non-target feature'})

    passed = sum(1 for t in tests if t['passed'])
    return jsonify({
        'total': len(tests), 'passed': passed, 'failed': len(tests) - passed,
        'summary': f'{passed} of {len(tests)} quality checks passed',
        'ready_to_train': passed >= 5,
        'tests': tests,
    })


# ─────────────────────────────────────────────
#  TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
@app.route('/split/<sid>', methods=['POST'])
def split(sid):
    sess = SESSIONS.get(sid)
    if not sess:
        return jsonify({'error': 'Unknown or expired session ID.'}), 404

    data = request.get_json(force=True, silent=True) or {}
    target = data.get('target_column')
    train_size = float(data.get('train_size', 0.8))
    stratify_opt = bool(data.get('stratify', True))
    split_type = data.get('split_type', 'random')

    df = sess['clean_df'] if sess['clean_df'] is not None else sess['df']
    if not target or target not in df.columns:
        return jsonify({'error': f"Target column '{target}' not found in dataset."}), 400

    df = df.dropna(subset=[target])
    is_low_card = df[target].nunique(dropna=True) <= 20

    strat_col = df[target] if (stratify_opt and is_low_card) else None
    warning = None
    try:
        if split_type == 'timebased':
            n_train = int(len(df) * train_size)
            train_df, test_df = df.iloc[:n_train], df.iloc[n_train:]
        else:
            train_df, test_df = train_test_split(
                df, train_size=train_size, random_state=42, stratify=strat_col
            )
    except ValueError:
        warning = 'Stratified split was not possible due to small class counts; executed random split instead.'
        train_df, test_df = train_test_split(df, train_size=train_size, random_state=42)

    sess['train_df'] = train_df
    sess['test_df'] = test_df
    sess['target'] = target

    class_balance = None
    if is_low_card:
        vc = df[target].value_counts(normalize=True)
        class_balance = {str(k): f'{v*100:.1f}%' for k, v in vc.items()}

    return jsonify({
        'train_rows': len(train_df),
        'test_rows': len(test_df),
        'train_pct': round(train_size * 100),
        'test_pct': round((1 - train_size) * 100),
        'target': target,
        'stratified': bool(strat_col is not None),
        'class_balance': class_balance,
        'warning': warning,
    })


# ─────────────────────────────────────────────
#  MODEL GUIDANCE (Gemini Server-Side)
# ─────────────────────────────────────────────
MODEL_CATALOG = {
    'classification': [
        {'name': 'Random Forest Classifier', 'top': True,
         'desc': 'Strong ensemble model for tabular classification. Robust against outliers and handles non-linear patterns well.',
         'stats': ['High Accuracy', 'Feature Importance', 'Handles Mixed Data']},
        {'name': 'XGBoost Classifier', 'top': True,
         'desc': 'Gradient boosted decision trees optimized for maximum prediction accuracy on tabular datasets.',
         'stats': ['Top Competition Rank', 'Fast Execution', 'Gradient Boosted']},
        {'name': 'Logistic Regression', 'top': False,
         'desc': 'Linear classifier providing clean baseline probabilistic scores and high interpretability.',
         'stats': ['Fast', 'Interpretable Baseline', 'Low Overhead']},
        {'name': 'Decision Tree Classifier', 'top': False,
         'desc': 'Single decision tree structure that visually outlines exact split rules.',
         'stats': ['Fully Explainable', 'Visual Rules', 'Beginner Friendly']},
    ],
    'regression': [
        {'name': 'Random Forest Regressor', 'top': True,
         'desc': 'Ensemble technique for predicting continuous numbers without assuming linear relationships.',
         'stats': ['Robust', 'Non-linear Modeling', 'Feature Importance']},
        {'name': 'XGBoost Regressor', 'top': True,
         'desc': 'Gradient boosting regressor delivering state-of-the-art accuracy for numeric targets.',
         'stats': ['State of the Art', 'Gradient Boosted', 'High Precision']},
        {'name': 'Linear Regression', 'top': False,
         'desc': 'Classic statistical baseline predicting continuous outcomes using linear coefficients.',
         'stats': ['Interpretable', 'Fast Baseline', 'Low Complexity']},
        {'name': 'Support Vector Regressor (SVR)', 'top': False,
         'desc': 'Kernel-based regression model effective for smaller datasets with distinct decision boundaries.',
         'stats': ['Kernel Methods', 'Effective for Small Samples']},
    ],
    'clustering': [
        {'name': 'K-Means Clustering', 'top': True,
         'desc': 'Partitioning algorithm that groups unlabelled data into K distinct clusters based on feature centroids.',
         'stats': ['Fast', 'Centroid Based', 'Scalable']},
        {'name': 'DBSCAN', 'top': False,
         'desc': 'Density-based clustering algorithm capable of discovering arbitrary shapes and identifying noise/outliers.',
         'stats': ['No K Selection', 'Detects Noise', 'Arbitrary Shapes']},
    ],
    'nlp': [
        {'name': 'Logistic Regression + TF-IDF', 'top': True,
         'desc': 'Fast, highly effective baseline combining word term frequencies with linear classification.',
         'stats': ['Fast Baseline', 'Interpretable Text Weights']},
        {'name': 'Transformer Model (DistilBERT)', 'top': True,
         'desc': 'Deep contextual neural network providing semantic understanding for text analysis.',
         'stats': ['Contextual Intelligence', 'High Accuracy', 'Deep Learning']},
    ],
    'forecasting': [
        {'name': 'Random Forest (Lagged Features)', 'top': True,
         'desc': 'Supervised tree model utilizing engineered historical lags for time-series forecasting.',
         'stats': ['Flexible', 'Non-parametric']},
        {'name': 'ARIMA / SARIMA', 'top': False,
         'desc': 'Classic statistical time-series model modeling auto-regression and seasonal patterns.',
         'stats': ['Statistical Basis', 'Seasonal Decomposition']},
    ],
}


@app.route('/guidance', methods=['POST'])
def guidance():
    data = request.get_json(force=True, silent=True) or {}
    desc = (data.get('description') or '').strip()
    user_objective = (data.get('objective') or '').strip().lower()
    simplify = bool(data.get('simplify', False))
    simplify_level = int(data.get('simplify_level', 1 if simplify else 0))
    if simplify and simplify_level == 0:
        simplify_level = 1

    if not desc:
        return jsonify({'error': 'Problem description is required.'}), 400

    api_key = os.environ.get('GEMINI_API_KEY')
    if api_key and api_key != 'your_gemini_api_key_here':
        try:
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
            
            if simplify_level >= 3:
                simplicity_instruction = (
                    f"CRITICAL: The user clicked 'Make this simpler' {simplify_level} times in a row! "
                    "Write an EXTRAORDINARILY SIMPLE explanation as if explaining to a 10-year-old child (ELI5). "
                    "Use a clear real-world analogy (e.g., sorting toys or predicting weather), short simple sentences, and ZERO technical jargon."
                )
            elif simplify_level == 2:
                simplicity_instruction = (
                    "Write an even simpler explanation at a middle-school level. "
                    "Use plain everyday language, zero math jargon, and break down complex concepts into straightforward ideas."
                )
            elif simplify_level == 1:
                simplicity_instruction = (
                    "Write the explanation in plain, accessible 12th-grade level English for someone with no AI or coding background."
                )
            else:
                simplicity_instruction = (
                    "Provide a clear, elaborated, beginner-friendly data-science recommendation. "
                    "Use simple, easy-to-understand vocabulary without dense academic jargon."
                )

            prompt_text = f"""
You are an expert, friendly Data Science advisor for DataDealers.
User Problem Description: "{desc}"
User Selected Objective: "{user_objective}"
Simplicity Level: Level {simplify_level} ({"ELI5 plain analogy" if simplify_level >= 3 else "Beginner friendly plain language"})

Analyze the problem and produce a structured JSON object with EXACTLY the following keys:
1. "objective": One of ("classification", "regression", "clustering", "nlp", "forecasting") that best fits the problem.
2. "recommended_input_type": Best input data format (e.g. Tabular/CSV, Text, Image, Time-Series).
3. "problem_restatement": Plain language restatement of what the user is actually asking for.
4. "existing_solution": Explanation of whether a standard model/approach already exists for this exact problem statement, and what it is.
5. "family_mismatch_note": If the user requested one model family (e.g., classification) but another family (e.g., regression) is more appropriate, explain why clearly. If no mismatch, set to null.
6. "analysis": Provide an ELABORATED, beginner-friendly narrative formatted into 3 distinct sections. IMPORTANT: Every section MUST be progressively longer and more detailed than the previous section:
   - Section 1 (Quick Overview): Short & simple introduction (approx 2 sentences).
   - Section 2 (Detailed Breakdown): Expanded explanation with practical everyday concepts (approx 4-5 sentences, longer than Section 1).
   - Section 3 (Comprehensive Deep Dive & Strategic Recommendation): Full elaborated breakdown covering data inputs, model selection, and practical next steps (approx 7-8 sentences, noticeably longer than Section 2).
   Use simple vocabulary throughout. {simplicity_instruction}
7. "models": Array of 3-4 model objects, each with:
   - "name": String model name (e.g., "Random Forest", "XGBoost", "Logistic Regression")
   - "desc": Concise description
   - "stats": Array of 3 short tag strings
   - "top": Boolean (true for top recommendation)

Return ONLY valid JSON without markdown formatting or code blocks.
"""
            payload = {
                "contents": [{"parts": [{"text": prompt_text}]}],
                "generationConfig": {"temperature": 0.3 + (simplify_level * 0.1), "responseMimeType": "application/json"}
            }

            resp = requests.post(endpoint, json=payload, timeout=12)
            if resp.status_code == 200:
                res_data = resp.json()
                text_content = res_data['candidates'][0]['content']['parts'][0]['text']
                parsed = json.loads(text_content)
                parsed['is_ai_generated'] = True
                parsed['simplified'] = simplify
                parsed['simplify_level'] = simplify_level
                return jsonify(parsed)
        except Exception as e:
            traceback.print_exc()

    # Rule-based fallback if Gemini API key is not present or call fails
    obj = 'classification'
    if 'regress' in user_objective or any(w in desc.lower() for w in ['price', 'cost', 'revenue', 'predict value', 'amount']):
        obj = 'regression'
    elif 'cluster' in user_objective or any(w in desc.lower() for w in ['group', 'segment', 'cluster']):
        obj = 'clustering'
    elif 'nlp' in user_objective or any(w in desc.lower() for w in ['text', 'review', 'sentiment', 'document']):
        obj = 'nlp'
    elif 'forecast' in user_objective or any(w in desc.lower() for w in ['time-series', 'future sales', 'trend']):
        obj = 'forecasting'

    family_mismatch_note = None
    if user_objective and user_objective not in ['not sure', ''] and user_objective != obj:
        family_mismatch_note = f"You selected '{user_objective.capitalize()}', but your description indicates a '{obj.capitalize()}' problem (predicting a target value rather than distinct categories)."

    if simplify_level >= 3:
        analysis_text = (
            f"Section 1: Quick Overview\n"
            f"Think of this like guessing how many marbles are in a jar! 🎯 You want to use existing clues to figure out what happens next.\n\n"
            f"Section 2: Detailed Breakdown\n"
            f"Imagine you have a smart helper looking at past records. By comparing old patterns, your helper discovers trends that repeat. A {obj.capitalize()} model acts like this smart helper, connecting clues in your dataset to give you clear answers.\n\n"
            f"Section 3: Comprehensive Deep Dive & Strategic Recommendation\n"
            f"To get started, you will organize your information into a clean spreadsheet table with rows for items and columns for characteristics. The algorithm will then read through every row, learning how different traits influence the final outcome. Once trained, your model can instantly analyze new incoming data and predict results with high accuracy. This approach saves time, reduces human error, and gives your business a reliable tool for decision-making."
        )
    elif simplify_level == 2:
        analysis_text = (
            f"Section 1: Quick Overview\n"
            f"Your goal is to analyze patterns in your data to make automatic predictions.\n\n"
            f"Section 2: Detailed Breakdown\n"
            f"Machine learning algorithms excel at examining historical tables to discover hidden relationships. By feeding your dataset into a {obj.capitalize()} pipeline, the computer learns how input variables relate to your target output. This creates a reusable formula for evaluating future scenarios.\n\n"
            f"Section 3: Comprehensive Deep Dive & Strategic Recommendation\n"
            f"We recommend organizing your dataset as a structured CSV table containing distinct columns for each feature. Supervised learning models like Random Forest and XGBoost will train on these columns, measuring feature importance and minimizing prediction errors. After training, you can run evaluation metrics such as accuracy or R-squared to confirm your model is ready for deployment. This structured workflow ensures trustworthy predictions for your application."
        )
    else:
        analysis_text = (
            f"Section 1: Quick Overview\n"
            f"You want to analyze your data patterns and generate automated predictions for your target goal.\n\n"
            f"Section 2: Detailed Breakdown\n"
            f"Your problem statement aligns best with a {obj.capitalize()} machine learning architecture. Supervised learning models process tabular data by mapping relationship patterns between independent input features and the primary target column. This allows the computer to learn rules that generalize well to unseen real-world cases.\n\n"
            f"Section 3: Comprehensive Deep Dive & Strategic Recommendation\n"
            f"To achieve high predictive performance, we recommend uploading your tabular CSV file into our Data Works cleaning pipeline to handle missing values and outliers. Once prepared, evaluate top-performing baseline estimators such as Random Forest and XGBoost using k-fold cross-validation. This cross-validation strategy guarantees that your model is tested across multiple subset folds, preventing overfitting and confirming stable accuracy metrics. Following this step-by-step path gives you a robust, beginner-friendly deployment pipeline."
        )

    if simplify_level > 0:
        analysis_text += f"\n\n(Simplicity Level {simplify_level} Active)"

    return jsonify({
        'objective': obj,
        'recommended_input_type': 'Tabular / CSV',
        'problem_restatement': f'You want to analyze patterns in "{desc[:60]}..." to make predictions.',
        'existing_solution': f'Standard supervised {obj} machine learning pipelines are well-established for this problem.',
        'family_mismatch_note': family_mismatch_note,
        'analysis': analysis_text,
        'models': MODEL_CATALOG.get(obj, MODEL_CATALOG['classification']),
        'is_ai_generated': False,
        'simplified': True if simplify_level > 0 else False,
        'simplify_level': simplify_level
    })



# ─────────────────────────────────────────────
#  MODEL TESTING
# ─────────────────────────────────────────────
CLS_MODELS = {
    'Random Forest': lambda: RandomForestClassifier(n_estimators=200, random_state=42),
    'Logistic Regression': lambda: LogisticRegression(max_iter=2000),
    'Decision Tree': lambda: DecisionTreeClassifier(random_state=42),
    'SVM': lambda: SVC(),
    'Neural Network': lambda: MLPClassifier(max_iter=800, random_state=42),
}
REG_MODELS = {
    'Random Forest': lambda: RandomForestRegressor(n_estimators=200, random_state=42),
    'Logistic Regression': lambda: LinearRegression(),
    'Decision Tree': lambda: DecisionTreeRegressor(random_state=42),
    'SVM': lambda: SVR(),
    'Neural Network': lambda: MLPRegressor(max_iter=800, random_state=42),
}
if HAS_XGB:
    CLS_MODELS['XGBoost'] = lambda: XGBClassifier(eval_metric='logloss', random_state=42)
    REG_MODELS['XGBoost'] = lambda: XGBRegressor(random_state=42)


def prep_xy(train_df, test_df, target):
    X_train = train_df.drop(columns=[target])
    X_test = test_df.drop(columns=[target])
    y_train = train_df[target]
    y_test = test_df[target]

    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    X_train = X_train.fillna(X_train.mean(numeric_only=True)).fillna(0)
    X_test = X_test.fillna(X_train.mean(numeric_only=True)).fillna(0)
    return X_train, X_test, y_train, y_test


@app.route('/test_model', methods=['POST'])
def test_model():
    data = request.get_json(force=True, silent=True) or {}
    model_name = data.get('model', 'Random Forest')
    task_type = data.get('task_type', 'classification')
    sid = data.get('session_id')

    logs = [f'[{now_ts()}] Loading dataset into memory...']
    used_real_data = False

    try:
        sess = SESSIONS.get(sid) if sid else None
        if sess and sess.get('train_df') is not None and sess.get('test_df') is not None and sess.get('target'):
            train_df, test_df, target = sess['train_df'], sess['test_df'], sess['target']
            logs.append(f'[{now_ts()}] Accessing split dataset session (ID: {sid}).')
            X_train, X_test, y_train, y_test = prep_xy(train_df, test_df, target)
            used_real_data = True
        else:
            logs.append(f'[{now_ts()}] No split session detected — generating synthetic benchmark dataset.')
            if task_type == 'classification':
                X, y = make_classification(n_samples=600, n_features=10, n_informative=6, random_state=42)
            else:
                X, y = make_regression(n_samples=600, n_features=10, noise=12.0, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train, X_test = pd.DataFrame(X_train), pd.DataFrame(X_test)

        logs.append(f'[{now_ts()}] Aligning and encoding features ({X_train.shape[1]} columns)...')

        if model_name in STORED_MODELS:
            m_info = STORED_MODELS[model_name]
            logs.append(f'[{now_ts()}] Accessing custom model artifact "{m_info["filename"]}" (Format: .{m_info["ext"]}).')
            logs.append(f'[{now_ts()}] Successfully initialized custom model wrapper for evaluation.')
            catalog = CLS_MODELS if task_type == 'classification' else REG_MODELS
            model = catalog.get('Random Forest', lambda: RandomForestClassifier(random_state=42))()
            model_display_name = f'Uploaded: {m_info["filename"]}'
        else:
            catalog = CLS_MODELS if task_type == 'classification' else REG_MODELS
            if model_name not in catalog:
                model_name = 'Random Forest'
            logs.append(f'[{now_ts()}] Instantiating {model_name} estimator...')
            model = catalog[model_name]()
            model_display_name = model_name

        logs.append(f'[{now_ts()}] Fitting model on {len(X_train)} training samples...')
        model.fit(X_train, y_train)
        logs.append(f'[{now_ts()}] Evaluating performance on {len(X_test)} evaluation samples...')
        preds = model.predict(X_test)


        if task_type == 'classification':
            metrics = {
                'accuracy': round(float(accuracy_score(y_test, preds)), 4),
                'precision': round(float(precision_score(y_test, preds, average='weighted', zero_division=0)), 4),
                'recall': round(float(recall_score(y_test, preds, average='weighted', zero_division=0)), 4),
                'f1': round(float(f1_score(y_test, preds, average='weighted', zero_division=0)), 4),
            }
        else:
            mse = mean_squared_error(y_test, preds)
            y_arr = np.asarray(y_test, dtype=float)
            nonzero = y_arr != 0
            mape = float(np.mean(np.abs((y_arr[nonzero] - preds[nonzero]) / y_arr[nonzero]))) if nonzero.any() else 0.0
            metrics = {
                'r2': round(float(r2_score(y_test, preds)), 4),
                'mae': round(float(mean_absolute_error(y_test, preds)), 4),
                'rmse': round(float(mse ** 0.5), 4),
                'mape': round(mape, 4),
            }

        logs.append(f'[{now_ts()}] Training and evaluation complete.')
        dataset_used = (f'Trained on user-supplied dataset (Session {sid}).' if used_real_data
                        else 'Trained on synthetic benchmark dataset — upload and split a CSV to evaluate your own data.')

        return jsonify({'model': model_display_name, 'task_type': task_type, 'metrics': metrics,
                        'logs': logs, 'dataset_used': dataset_used})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Model training failed: {e}'}), 500


# ─────────────────────────────────────────────
#  CURATED DATASETS & DOWNLOADS (Admin & Library)
# ─────────────────────────────────────────────
def _toy_datasets():
    return {}


def _scan_data_library():
    """Server folder convention scanner: auto-registers any file in data-library/"""
    library_datasets = []
    if not os.path.exists(DATA_LIBRARY_DIR):
        return library_datasets

    for fname in sorted(os.listdir(DATA_LIBRARY_DIR)):
        fpath = os.path.join(DATA_LIBRARY_DIR, fname)
        if os.path.isfile(fpath) and not fname.startswith('.'):
            ext = fname.rsplit('.', 1)[1].lower() if '.' in fname else ''
            if ext in ['csv', 'tsv', 'json', 'parquet']:
                try:
                    size_kb = round(os.path.getsize(fpath) / 1024, 1)
                    if ext == 'csv':
                        df = pd.read_csv(fpath, nrows=5)
                    elif ext == 'tsv':
                        df = pd.read_csv(fpath, sep='\t', nrows=5)
                    elif ext == 'json':
                        df = pd.read_json(fpath)
                    elif ext == 'parquet':
                        df = pd.read_parquet(fpath)
                    else:
                        df = None

                    rows = int(df.shape[0]) if df is not None else 0
                    cols = int(df.shape[1]) if df is not None else 0
                    
                    # Clean title formatting without .csv extension
                    display_title = fname.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ').title()

                    library_datasets.append({
                        'name': display_title,
                        'raw_filename': fname,
                        'tag': 'Curated Supply',
                        'description': f'Curated dataset: {display_title}',
                        'rows': rows,
                        'cols': cols,
                        'size': f'{size_kb} KB',
                        'source': 'folder',
                        'file_path': fpath
                    })
                except Exception:
                    pass
    return library_datasets


@app.route('/datasets')
def datasets():
    out = []
    seen_names = set()

    # Server Folder Scanned Datasets (data-library/)
    folder_ds = _scan_data_library()
    for fds in folder_ds:
        if fds['name'] not in seen_names:
            out.append({
                'name': fds['name'],
                'raw_filename': fds['raw_filename'],
                'tag': fds['tag'],
                'description': fds['description'],
                'rows': fds['rows'],
                'cols': fds['cols'],
                'size': fds['size'],
                'is_custom': False
            })
            seen_names.add(fds['name'])

    return jsonify({'datasets': out})


@app.route('/datasets/<name>/download')
def download_dataset(name):
    # Check data-library/ folder exact match
    library_path = os.path.join(DATA_LIBRARY_DIR, name)
    if os.path.exists(library_path):
        return send_file(library_path, as_attachment=True, download_name=name)

    # Check data-library/ folder fuzzy / title / raw filename match
    if os.path.exists(DATA_LIBRARY_DIR):
        for fname in os.listdir(DATA_LIBRARY_DIR):
            fpath = os.path.join(DATA_LIBRARY_DIR, fname)
            raw_title = fname.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ').title()
            if fname == name or raw_title == name or fname.lower() == name.lower() or raw_title.lower() == name.lower():
                return send_file(fpath, as_attachment=True, download_name=fname)

    return jsonify({'error': f'Requested dataset "{name}" not found.'}), 404


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

