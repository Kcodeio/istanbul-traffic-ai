# Istanbul Traffic AI 🚦🗺️

An experimental **traffic risk prediction** and **decision-support** application for Istanbul.

This project estimates **traffic congestion risk** for a selected **district**, **day**, and **hour**, and provides a simple, human-readable decision:

> **Is it a good idea to go out?**

The main goal is to demonstrate a **complete end-to-end ML workflow** combined with a practical UI and a lightweight decision engine — not to provide real-world traffic guidance.

🚀 **Live Demo:**  
https://istanbul-traffic-ai-kcodeio.streamlit.app/

---

## 🚀 Features

### 🚦 Traffic Risk Prediction
- Predicts traffic congestion risk as a **percentage**
- Based on:
  - District
  - Day of week
  - Hour of day
- Uses a trained **RandomForest** model

---

### 🧠 Decision Engine
- Converts numeric risk into a clear **YES / NO** recommendation
- Explains *why* the decision was made in plain English
- Keeps the logic transparent and interpretable

---

### 🗺️ Traffic Map (Visual Aid)
- Displays predicted traffic risk for **all districts**
- Color-coded markers:
  - 🟢 Low traffic
  - 🟡 Moderate traffic
  - 🔴 High traffic
- The map is an **auxiliary visualization**, not the main decision source

---

### 📊 Synthetic Dataset
- Automatically generated (~**180,000+ rows**)
- Encodes realistic assumptions:
  - Morning & evening rush hours
  - Weekdays vs weekends
  - Late-night traffic drops
  - District-specific profiles:
    - Central business areas
    - Residential zones
    - Highway-heavy districts

---

### ⚙️ Self-Contained Training
- If the dataset or model does not exist:
  - The dataset is generated automatically
  - The model is trained automatically
- No manual preprocessing required
- Fully reproducible pipeline

---

## 🧠 How It Works

1. A synthetic dataset is generated using predefined traffic heuristics
2. A `RandomForestClassifier` learns congestion patterns
3. The user selects:
   - District
   - Day of week
   - Hour
4. The model predicts traffic risk
5. A rule-based decision layer interprets the prediction
6. Results are shown together with an optional map visualization

---

## ⚠️ Disclaimer

**This project does NOT use real-time or official traffic data.**

- All data is **synthetic**
- Predictions are **not authoritative**
- The project is intended strictly for:
  - Learning
  - Experimentation
  - Demonstration of ML + UI integration

🚫 **Do NOT use this project for navigation, safety, or real-world decision-making.**

---

## 🛠️ Tech Stack

- Python 3
- Streamlit
- scikit-learn
- pandas
- numpy
- folium
- streamlit-folium
- joblib

---

## 🖼️ Screenshots

<img width="1919" height="875" alt="Traffic prediction UI" src="https://github.com/user-attachments/assets/9827dcfe-47e7-463d-9e8f-76b460173b95" />

<img width="1919" height="872" alt="Traffic map visualization" src="https://github.com/user-attachments/assets/c8e2fba8-f503-4f25-82b7-37699c85fec7" />

![Live demo interaction](https://github.com/user-attachments/assets/2fd1ab54-8be4-4283-8865-feac557ae30f)

---
## ⭐ Support

If you find this project interesting or useful, consider giving it a star ⭐
It helps visibility and motivates further improvements.

## ▶️ Running the App Locally

```bash
pip install streamlit numpy pandas scikit-learn folium streamlit-folium joblib
streamlit run app.py
