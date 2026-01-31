# ============================================================
# Istanbul Traffic AI 🚦🗺️
# FINAL – Decision + Reasons + Confidence (Separated) + Map
# ============================================================

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier

import streamlit as st
import folium
from streamlit_folium import st_folium

# ===================== SESSION STATE =====================
if "predicted" not in st.session_state:
    st.session_state.predicted = False
if "result" not in st.session_state:
    st.session_state.result = None

# ===================== PATHS =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "traffic.csv")
VEC_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

# ===================== DISTRICTS =====================
DISTRICTS = {
    "Besiktas": (41.0430, 29.0094),
    "Kadikoy": (40.9917, 29.0275),
    "Sisli": (41.0602, 28.9877),
    "Beyoglu": (41.0370, 28.9770),
    "Fatih": (41.0186, 28.9399),
    "Uskudar": (41.0230, 29.0157),
    "Eyup": (41.0486, 28.9336),
    "E5": (41.0000, 28.9000),
    "TEM": (41.1000, 28.8000),
}

CORE = {"Besiktas", "Kadikoy", "Sisli", "Beyoglu", "Fatih", "Uskudar"}

# ===================== HELPERS =====================
def hour_cyclic(h):
    a = 2 * np.pi * h / 24
    return np.sin(a), np.cos(a)

def day_cyclic(d):
    a = 2 * np.pi * d / 7
    return np.sin(a), np.cos(a)

def risk_color(r):
    if r >= 0.7:
        return "red"
    if r >= 0.4:
        return "orange"
    return "green"

# ===================== DATA =====================
def generate_dataset():
    rows = []
    for dow in range(7):
        for hour in range(24):
            for district in DISTRICTS:
                base = 0.6 if district in CORE else 0.35
                if dow >= 5:
                    base -= 0.25
                if hour <= 5 or hour >= 23:
                    base -= 0.25
                if 7 <= hour <= 10 or 17 <= hour <= 20:
                    base += 0.35

                hs, hc = hour_cyclic(hour)
                ds, dc = day_cyclic(dow)

                rows.append({
                    "dow": dow,
                    "hour": hour,
                    "district": district,
                    "is_weekend": int(dow >= 5),
                    "rush_morning": int(7 <= hour <= 10),
                    "rush_evening": int(17 <= hour <= 20),
                    "is_night": int(hour <= 5 or hour >= 23),
                    "hour_sin": hs,
                    "hour_cos": hc,
                    "dow_sin": ds,
                    "dow_cos": dc,
                    "y": int(base >= 0.5)
                })
    df = pd.DataFrame(rows)
    df.to_csv(DATA_PATH, index=False)
    return df

def load_or_train():
    if os.path.exists(VEC_PATH) and os.path.exists(MODEL_PATH):
        return joblib.load(VEC_PATH), joblib.load(MODEL_PATH)

    df = generate_dataset()
    X_cols = [c for c in df.columns if c != "y"]
    vec = DictVectorizer()
    X = vec.fit_transform(df[X_cols].to_dict("records"))

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=16,
        random_state=42
    )
    model.fit(X, df["y"])

    joblib.dump(vec, VEC_PATH)
    joblib.dump(model, MODEL_PATH)
    return vec, model

# ===================== DECISION + CONFIDENCE =====================
def decision_and_confidence(risk, hour, dow):
    decision_reasons = []
    confidence_reasons = []
    confidence_score = 0

    # --- Human decision reasons ---
    if risk < 0.35:
        decision_reasons.append("Traffic levels seem manageable.")
    elif risk < 0.6:
        decision_reasons.append("Traffic conditions may cause some delays.")
    else:
        decision_reasons.append("Heavy traffic is expected.")

    if 7 <= hour <= 10 or 17 <= hour <= 20:
        decision_reasons.append("This is a peak rush hour.")
    else:
        decision_reasons.append("It is outside peak hours.")

    if dow >= 5:
        decision_reasons.append("It is a weekend, which usually reduces traffic pressure.")

    if hour <= 5 or hour >= 23:
        decision_reasons.append("Late-night hours usually have lighter traffic.")

    # --- Confidence logic (model-side) ---
    if risk <= 0.15 or risk >= 0.75:
        confidence_score += 2
        confidence_reasons.append("The model shows strong agreement for this risk level.")
    elif 0.4 <= risk <= 0.6:
        confidence_reasons.append("The predicted risk is close to the decision boundary.")

    if 7 <= hour <= 10 or 17 <= hour <= 20:
        confidence_score += 1
        confidence_reasons.append("This time corresponds to well-known traffic patterns.")

    if dow >= 5:
        confidence_score += 1
        confidence_reasons.append("Weekend traffic behavior is consistently represented in the data.")

    if dow >= 5 and (7 <= hour <= 10 or 17 <= hour <= 20):
        confidence_score -= 1
        confidence_reasons.append("Weekend and rush hour signals partially conflict.")

    if confidence_score >= 3:
        confidence = "High"
    elif confidence_score >= 1:
        confidence = "Medium"
    else:
        confidence = "Low"

    # --- Final decision ---
    if risk >= 0.6:
        decision = "NO"
    elif risk <= 0.35:
        decision = "YES"
    else:
        decision = "NO" if confidence == "Low" else "YES"

    return decision, decision_reasons, confidence, confidence_reasons

# ===================== UI =====================
st.set_page_config("Istanbul Traffic AI 🚦🗺️", layout="wide")
st.title("Istanbul Traffic AI 🚦🗺️")

vec, model = load_or_train()

left, right = st.columns([1, 2])

with left:
    district = st.selectbox("District", list(DISTRICTS.keys()))
    dow = st.selectbox("Day (0 = Monday)", list(range(7)))
    hour = st.slider("Hour", 0, 23, 18)

    if st.button("Predict", type="primary"):
        st.session_state.predicted = True

        hs, hc = hour_cyclic(hour)
        ds, dc = day_cyclic(dow)

        features = {
            "dow": dow,
            "hour": hour,
            "district": district,
            "is_weekend": int(dow >= 5),
            "rush_morning": int(7 <= hour <= 10),
            "rush_evening": int(17 <= hour <= 20),
            "is_night": int(hour <= 5 or hour >= 23),
            "hour_sin": hs,
            "hour_cos": hc,
            "dow_sin": ds,
            "dow_cos": dc,
        }

        risk = float(model.predict_proba(vec.transform([features]))[0, 1])
        risk = max(risk, 0.03)

        decision, d_reasons, confidence, c_reasons = decision_and_confidence(risk, hour, dow)

        st.session_state.result = {
            "risk": risk,
            "decision": decision,
            "decision_reasons": d_reasons,
            "confidence": confidence,
            "confidence_reasons": c_reasons,
            "features": features
        }

with right:
    if st.session_state.predicted and st.session_state.result:
        res = st.session_state.result

        st.metric("Traffic Risk", f"{res['risk']*100:.1f}%")

        st.subheader("Is it a good idea to go out?")
        if res["decision"] == "YES":
            st.success(f"YES")
        else:
            st.error(f"NO")

        st.subheader("Why this recommendation?")
        for r in res["decision_reasons"]:
            st.write(f"- {r}")

        st.subheader("Confidence")
        if res["confidence"] == "High":
            st.success(f"{res['confidence']}")
        elif res["confidence"] == "Medium":
            st.warning(f"{res['confidence']}")
        else:
            st.error(f"{res['confidence']}")

        st.subheader("Why the model is confident?")
        for r in res["confidence_reasons"]:
            st.write(f"- {r}")

        st.divider()
        st.subheader("Traffic Map (visual aid)")

        m = folium.Map(location=[41.02, 28.98], zoom_start=11, tiles="CartoDB positron")
        for name, (lat, lon) in DISTRICTS.items():
            f = dict(res["features"])
            f["district"] = name
            r = float(model.predict_proba(vec.transform([f]))[0, 1])

            folium.CircleMarker(
                location=[lat, lon],
                radius=24 if name == district else 16,
                color=risk_color(r),
                fill=True,
                fill_color=risk_color(r),
                fill_opacity=0.75,
                popup=f"{name}: {r*100:.1f}%"
            ).add_to(m)

        st_folium(m, width=850, height=520)
    else:
        st.info("Select inputs and press **Predict** to see results.")
