import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath

import joblib
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer

from sklearn.pipeline import Pipeline
import shap


# ── Setup & Path Configuration ─────────────────────────────────────────────────
warnings.simplefilter("ignore")

# On Streamlit Cloud the script lives inside the repo, e.g.:
#   /mount/src/stock_prediction-/Portfolio/StreamlitApp_Week_7.py
# We need BOTH the current dir AND its parent on sys.path so that
#   from src.feature_utils import ...   resolves correctly.
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

for p in [current_dir, project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.feature_utils import extract_features_pair

# ── Secrets ────────────────────────────────────────────────────────────────────
aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# ── AWS Session ────────────────────────────────────────────────────────────────
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session    = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# ── Data & Model Configuration ─────────────────────────────────────────────────
df_features = extract_features_pair()

# FIX: "keys" and "inputs" must reference the SAME two tickers (LHX pair).
#      Previously "keys" said ["LHX","INCY"] while "inputs" said ["LHX","ACN"] — mismatch.
#      Update the second ticker to match whichever valid_partner was found in the notebook.
PAIR_TICKER  = "LHX"
PAIR_PARTNER = df_features.columns[df_features.columns != PAIR_TICKER][0]  # auto-detect partner

MODEL_INFO = {
    "endpoint": aws_endpoint,
    "explainer": 'explainer_pair.shap',
    "pipeline":  'finalized_pair_model.tar.gz',
    "keys":   [PAIR_PARTNER, PAIR_TICKER],   # must match column order in df_features
    "inputs": [
        {"name": PAIR_PARTNER, "type": "number", "min": 0.0, "default": 0.0, "step": 10.0},
        {"name": PAIR_TICKER,  "type": "number", "min": 0.0, "default": 0.0, "step": 10.0},
    ]
}

# ── Model / Explainer Loaders ──────────────────────────────────────────────────
@st.cache_resource
def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename  = MODEL_INFO["pipeline"]

    s3_client.download_file(
        Filename=filename,
        Bucket=bucket,
        Key=f"{key}/{os.path.basename(filename)}"
    )

    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]

    return joblib.load(joblib_file)


@st.cache_resource
def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')

    if not os.path.exists(local_path):
        # FIX: download_file uses Filename= for local dest, Key= for S3 object
        s3_client.download_file(Bucket=bucket, Key=key, Filename=local_path)

    with open(local_path, "rb") as f:
        return shap.Explainer.load(f)

# ── Prediction ─────────────────────────────────────────────────────────────────
def call_model_api(input_df):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer()
    )
    try:
        raw_pred = predictor.predict(input_df.values.astype(np.float64))
        pred_val = pd.DataFrame(raw_pred).values[-1][0]
        return round(float(pred_val), 4), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

# ── Local Explainability ───────────────────────────────────────────────────────
def display_explanation(input_df, _session, bucket):
    explainer_name = MODEL_INFO["explainer"]
    local_path     = os.path.join(tempfile.gettempdir(), explainer_name)

    explainer = load_shap_explainer(
        _session,
        bucket,
        posixpath.join('explainer', explainer_name),
        local_path
    )

    best_pipeline        = load_pipeline(_session, bucket, 'sklearn-pipeline-deployment')
    preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-2])
    input_df_transformed = preprocessing_pipeline.transform(input_df)
    feature_names        = best_pipeline[1:4].get_feature_names_out()
    input_df_transformed = pd.DataFrame(input_df_transformed, columns=feature_names)

    shap_values = explainer(input_df_transformed)   # FIX: use transformed df, not raw input_df

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0, :, 0], max_display=10, show=False)
    st.pyplot(fig)

    # FIX: correct attribute access — was `.values` (invalid); correct is just the Series
    shap_series = pd.Series(
        shap_values[0, :, 0].values,
        index=shap_values[0, :, 0].feature_names
    )
    top_feature = shap_series.abs().idxmax()
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")

# ── Streamlit UI ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="ML Deployment — LHX Pairs Trading", layout="wide")
st.title("👨‍💻 ML Deployment — LHX Pairs Trading")

with st.form("pred_form"):
    st.subheader("Inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['name'].replace('_', ' ').upper(),
                min_value=inp['min'],
                value=inp['default'],
                step=inp['step']
            )

    submitted = st.form_submit_button("Run Prediction")

if submitted:
    data_row = [user_inputs[k] for k in MODEL_INFO["keys"]]

    base_df  = df_features
    input_df = pd.concat(
        [base_df, pd.DataFrame([data_row], columns=base_df.columns)],
        ignore_index=True
    )

    res, status = call_model_api(input_df)

    if status == 200:
        signal_map = {1: "📈 BUY", 0: "⏸ HOLD", -1: "📉 SELL"}
        label = signal_map.get(int(res), str(res))
        st.metric("Prediction Result", label)
        display_explanation(input_df, session, aws_bucket)
    else:
        st.error(res)
