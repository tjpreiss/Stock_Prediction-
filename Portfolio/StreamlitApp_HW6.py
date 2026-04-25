import os, sys, warnings, tarfile, tempfile, posixpath
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer
from sklearn.pipeline import Pipeline
import shap

warnings.simplefilter("ignore")

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AMZN Return Predictor",
    page_icon="📈",
    layout="wide"
)

# ── AWS Secrets (set in Streamlit Cloud → Settings → Secrets) ─────────────────
# Format of .streamlit/secrets.toml:
# [aws_credentials]
# AWS_ACCESS_KEY_ID     = "..."
# AWS_SECRET_ACCESS_KEY = "..."
# AWS_SESSION_TOKEN     = ""        # leave blank if using long-term keys
# AWS_BUCKET            = "thomas-preiss-s3-bucket"
# AWS_ENDPOINT          = "amzn-xgboost-endpoint-1"

try:
    aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
    aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
    aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
    aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]
    aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]
except Exception:
    # Fallback defaults for local testing (no real AWS calls will succeed)
    aws_id = aws_secret = aws_token = ""
    aws_bucket   = "thomas-preiss-s3-bucket"
    aws_endpoint = "amzn-xgboost-endpoint-1"
    st.warning("⚠️  AWS secrets not found. Running in local/demo mode.")

TICKER = "AMZN"

# ── Model configuration ────────────────────────────────────────────────────────
# These must match the features used when training the best model.
# Update this list to whichever feature set your best model uses.
MODEL_INFO = {
    "endpoint":  aws_endpoint,
    "explainer": "explainer_sentiment.shap",
    "pipeline":  "finalized_sentiment_model.tar.gz",
    # Feature keys shown in the UI — match the columns fed to the deployed model.
    "keys": [
        "sentiment_textblob",
        "sentiment_LSTM",
        "sentiment_lex",
        "sentiment_textblob_lag1",
        "sentiment_LSTM_lag1",
        "sentiment_lex_lag1",
        "sentiment_textblob_lag2",
        "sentiment_LSTM_lag2",
        "sentiment_lex_lag2",
    ],
    "inputs": [
        {"name": "sentiment_textblob",     "min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01},
        {"name": "sentiment_LSTM",         "min":  0.0, "max": 1.0, "default": 0.5, "step": 0.01},
        {"name": "sentiment_lex",          "min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01},
        {"name": "sentiment_textblob_lag1","min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01},
        {"name": "sentiment_LSTM_lag1",    "min":  0.0, "max": 1.0, "default": 0.5, "step": 0.01},
        {"name": "sentiment_lex_lag1",     "min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01},
        {"name": "sentiment_textblob_lag2","min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01},
        {"name": "sentiment_LSTM_lag2",    "min":  0.0, "max": 1.0, "default": 0.5, "step": 0.01},
        {"name": "sentiment_lex_lag2",     "min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01},
    ],
}


# ── AWS Session ────────────────────────────────────────────────────────────────
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    kwargs = dict(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        region_name="us-east-1",
    )
    if aws_token:
        kwargs["aws_session_token"] = aws_token
    return boto3.Session(**kwargs)


session    = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)


# ── S3 Loaders ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Downloading model pipeline from S3…")
def load_pipeline(_session, bucket, s3_key):
    s3_client = _session.client("s3")
    filename  = MODEL_INFO["pipeline"]
    local_tar = os.path.join(tempfile.gettempdir(), filename)
    s3_client.download_file(Bucket=bucket, Key=f"{s3_key}/{filename}", Filename=local_tar)
    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(path=tempfile.gettempdir())
        joblib_file = [f for f in tar.getnames() if f.endswith(".joblib")][0]
    return joblib.load(os.path.join(tempfile.gettempdir(), joblib_file))


@st.cache_resource(show_spinner="Downloading SHAP explainer from S3…")
def load_shap_explainer(_session, bucket, s3_key):
    s3_client  = _session.client("s3")
    local_path = os.path.join(tempfile.gettempdir(), MODEL_INFO["explainer"])
    if not os.path.exists(local_path):
        s3_client.download_file(
            Bucket=bucket,
            Key=posixpath.join("explainer", MODEL_INFO["explainer"]),
            Filename=local_path,
        )
    return joblib.load(local_path)


# ── Prediction via SageMaker Endpoint ─────────────────────────────────────────
def call_model_api(input_df: pd.DataFrame):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer(),
    )
    try:
        raw_pred = predictor.predict(input_df.values.astype(np.float32))
        pred_val = float(np.array(raw_pred).ravel()[0])
        return pred_val, 200
    except Exception as e:
        return str(e), 500


# ── SHAP Explanation ──────────────────────────────────────────────────────────
def display_explanation(input_df: pd.DataFrame):
    try:
        pipeline_obj  = load_pipeline(session, aws_bucket, "sklearn-pipeline-deployment")
        shap_explainer = load_shap_explainer(session, aws_bucket, "explainer")

        # Apply preprocessing (all steps except the final model)
        preprocessing    = Pipeline(pipeline_obj.steps[:-1])
        X_transformed    = preprocessing.transform(input_df)

        try:
            feature_names = pipeline_obj[:-1].get_feature_names_out()
        except Exception:
            feature_names = MODEL_INFO["keys"]

        X_df          = pd.DataFrame(X_transformed, columns=feature_names[:X_transformed.shape[1]])
        shap_values   = shap_explainer(X_df)

        st.subheader("🔍 Decision Transparency (SHAP Waterfall)")
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig, use_container_width=True)

        # Most influential feature
        try:
            top_feature = (
                pd.Series(shap_values[0].values, index=shap_values[0].feature_names)
                .abs()
                .idxmax()
            )
            st.info(f"**Key driver of this prediction:** `{top_feature}`")
        except Exception:
            pass

    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title(f"📈 {TICKER} Stock Return Predictor")
st.markdown(
    """
    Predict the **next-day return** of **Amazon (AMZN)** using news-sentiment scores
    (TextBlob, LSTM, and Lexicon) from the deployed SageMaker endpoint.

    Enter today's sentiment values and their 1- and 2-day lags, then click **Run Prediction**.
    """
)

st.divider()

with st.form("pred_form"):
    st.subheader("📝 Sentiment Feature Inputs")
    st.caption(
        "Sentiment range: TextBlob / Lexicon → [−1, 1] · LSTM → [0, 1] (probability of positive)"
    )
    cols = st.columns(3)
    user_inputs: dict = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 3]:
            label = inp["name"].replace("_", " ").upper()
            user_inputs[inp["name"]] = st.number_input(
                label,
                min_value=float(inp["min"]),
                max_value=float(inp["max"]),
                value=float(inp["default"]),
                step=float(inp["step"]),
                format="%.3f",
            )

    submitted = st.form_submit_button("🚀 Run Prediction", use_container_width=True)

# ── Results ───────────────────────────────────────────────────────────────────
if submitted:
    data_row = [user_inputs[k] for k in MODEL_INFO["keys"]]
    input_df = pd.DataFrame([data_row], columns=MODEL_INFO["keys"])

    with st.spinner("Calling SageMaker endpoint…"):
        result, status = call_model_api(input_df)

    if status == 200:
        pred_pct = result * 100
        direction = "📈 UP" if result > 0 else "📉 DOWN"
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label=f"{TICKER} Predicted Next-Day Return",
                value=f"{pred_pct:+.4f}%",
                delta=direction,
            )
        with col2:
            st.metric(
                label="Raw Prediction",
                value=f"{result:.6f}",
                help="Raw model output (fractional daily return)",
            )

        st.divider()
        display_explanation(input_df)

    else:
        st.error(f"Prediction failed (status {status}):\n{result}")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ App Info")
    st.markdown(f"""
    | | |
    |---|---|
    | **Target** | `{TICKER}` |
    | **S3 Bucket** | `{aws_bucket}` |
    | **Endpoint** | `{aws_endpoint}` |
    | **Model** | XGBoost Regressor |
    | **Features** | Sentiment (TextBlob + LSTM + Lexicon) + lags |
    """)

    st.divider()
    st.subheader("📖 How to Set Secrets")
    st.code("""
# .streamlit/secrets.toml
[aws_credentials]
AWS_ACCESS_KEY_ID     = "AKIA..."
AWS_SECRET_ACCESS_KEY = "..."
AWS_SESSION_TOKEN     = ""
AWS_BUCKET            = "thomas-preiss-s3-bucket"
AWS_ENDPOINT          = "amzn-xgboost-endpoint-1"
    """, language="toml")

    st.divider()
    st.caption("HW6 · AMZN Return Prediction · thomas-preiss-s3-bucket")
