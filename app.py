# ============================================================
# Istanbul Traffic AI v5.2
# Predict + Decision + OPTIONAL MAP (FIXED)
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


# ===================== PATHS =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
for d in (DATA_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "traffic.csv")
VEC_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")


# ===================== DISTRICTS =====================
DISTRICTS = {
    "Besiktas": (41.0430, 29.0094),
    "Sisli": (41.0602, 28.9877),
    "Beyoglu": (41.0370, 28.9770),
    "Fatih": (41.0186, 28.9399),
    "Kadikoy": (40.9917, 29.0275),
    "Uskudar": (41.0230, 29.0157),
    "Eyup": (41.0486, 28.9336),
    "Gaziosmanpasa": (41.0585, 28.9150),
    "Maltepe": (40.9350, 29.1317),
    "Kartal": (40.9000, 29.1850),
    "Pendik": (40.8775, 29.2360),
    "E5": (41.0000, 28.9000),
    "TEM": (41.1000, 28.8000),
}


# ===================== HELPERS =====================
def hour_cyclic(h):
    a = 2 * np.pi * h / 24
    return float(np.sin(a)), float(np.cos(a))

def day_cyclic(d):
    a = 2 * np.pi * d / 7
    return float(np.sin(a)), float(np.cos(a))

def risk_color(r):
    if r >= 0.75: return "red"
    if r >= 0.40: return "orange"
    return "green"

def go_out_decision(risk, hour, dow):
    score=0; yes=[]; no=[]
    if risk>=0.75:
        score+=2; no.append("Traffic congestion is expected to be very high.")
    elif risk>=0.40:
        score+=0.8; no.append("Traffic may be moderate; expect delays.")
    else:
        yes.append("Traffic levels seem manageable.")

    if 7<=hour<=10 or 17<=hour<=20:
        score+=1.5; no.append("This is a peak rush hour.")
    else:
        yes.append("It is outside peak hours.")

    if dow>=5:
        yes.append("It is a weekend, which usually reduces traffic pressure.")

    return ("NO", no) if score>=2.5 else ("YES", yes)


# ===================== DATA & MODEL =====================
def generate_dataset():
    rows=[]
    for dow in range(7):
        for hour in range(24):
            for name in DISTRICTS:
                hs,hc=hour_cyclic(hour)
                ds,dc=day_cyclic(dow)
                base = 0.7 if name in ("Besiktas","Sisli","Beyoglu","E5","TEM") else 0.4
                if dow>=5: base-=0.3
                if hour>=23 or hour<=5: base-=0.3
                rows.append({
                    "dow":dow,"hour":hour,"district":name,
                    "is_weekend":int(dow>=5),
                    "rush_morning":int(7<=hour<=10),
                    "rush_evening":int(17<=hour<=20),
                    "is_night":int(hour<=5 or hour>=23),
                    "rainy":0,"accident":0,
                    "hour_sin":hs,"hour_cos":hc,
                    "dow_sin":ds,"dow_cos":dc,
                    "y":int(base>=0.5)
                })
    df=pd.DataFrame(rows)
    df.to_csv(DATA_PATH,index=False)
    return df

def ensure_model():
    if os.path.exists(VEC_PATH) and os.path.exists(MODEL_PATH):
        return joblib.load(VEC_PATH), joblib.load(MODEL_PATH)
    df = generate_dataset()
    feats=[c for c in df.columns if c!="y"]
    vec=DictVectorizer()
    X=vec.fit_transform(df[feats].to_dict("records"))
    model=RandomForestClassifier(n_estimators=80,random_state=42)
    model.fit(X,df["y"])
    joblib.dump(vec,VEC_PATH)
    joblib.dump(model,MODEL_PATH)
    return vec, model


# ===================== UI =====================
st.set_page_config("Istanbul Traffic AI", layout="wide")
st.title("The İstanbul Traffic AI for long trips - TITAFLT 🚦🗺️")


st.markdown("Powered by TITAFLT AI")
vec, model = ensure_model()

if "predicted" not in st.session_state:
    st.session_state["predicted"]=False

left,right=st.columns([1,2])

with left:
    district=st.selectbox("District",list(DISTRICTS.keys()))
    dow=st.selectbox("Day (0 = Monday)",list(range(7)))
    hour=st.slider("Hour",0,23,18)
    if st.button("Predict",type="primary"):
        st.session_state["predicted"]=True

with right:
    if st.session_state["predicted"]:
        # ---- Prediction ----
        feat={
            "dow":dow,"hour":hour,"district":district,
            "is_weekend":int(dow>=5),
            "rush_morning":int(7<=hour<=10),
            "rush_evening":int(17<=hour<=20),
            "is_night":int(hour<=5 or hour>=23),
            "rainy":0,"accident":0,
            **dict(zip(
                ["hour_sin","hour_cos","dow_sin","dow_cos"],
                (*hour_cyclic(hour),*day_cyclic(dow))
            ))
        }
        risk=float(model.predict_proba(vec.transform([feat]))[0,1])
        st.metric("Traffic Risk",f"{risk*100:.1f}%")

        dec,reasons=go_out_decision(risk,hour,dow)
        st.subheader("Is it a good idea to go out?")
        st.markdown(f"## **{dec}**")
        for r in reasons: st.write(f"- {r}")

        # ---- MAP (EXTRA FEATURE) ----
        st.divider()
        st.subheader("Traffic Map (visual aid)")
        m=folium.Map(location=[41.015,28.98],zoom_start=11,tiles="CartoDB positron")
        for name,(lat,lon) in DISTRICTS.items():
            f=dict(feat); f["district"]=name
            r=float(model.predict_proba(vec.transform([f]))[0,1])
            folium.CircleMarker(
                location=[lat,lon],
                radius=26 if name==district else 18,
                color=risk_color(r),
                fill=True,fill_color=risk_color(r),fill_opacity=0.75,
                popup=f"{name} – {r*100:.1f}%"
            ).add_to(m)
        st_folium(m,width=800,height=520)
    else:
        st.info("Select inputs and press **Predict**. The map will appear after prediction.")
