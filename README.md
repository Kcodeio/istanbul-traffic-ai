# Istanbul Traffic AI 🚦🗺️

An experimental traffic risk prediction and decision-support application for Istanbul.

This project estimates **traffic congestion risk** for a selected district, day, and hour, and provides a simple decision:

> **Is it a good idea to go out?**

The focus of the project is to demonstrate a **complete ML workflow** combined with a practical UI and a lightweight decision engine.

---

## 🚀 Features

- 🚦 **Traffic Risk Prediction**
  - Predicts congestion risk as a percentage
  - Based on district, day of week, and hour
  - Uses a trained RandomForest model

- 🧠 **Decision Engine**
  - Converts the predicted risk into a clear **YES / NO** recommendation
  - Explains the reasoning in plain English

- 🗺️ **Traffic Map (Visual Aid)**
  - Displays predicted traffic risk for **all districts**
  - Color-coded markers:
    - 🟢 Low traffic
    - 🟡 Moderate traffic
    - 🔴 High traffic
  - The map is an **additional visualization**, not the main decision source

- 📊 **Synthetic Dataset**
  - Automatically generated (≈180k+ rows)
  - Encodes realistic assumptions:
    - Rush hours
    - Weekdays vs weekends
    - Late-night traffic
    - District-specific profiles (core, residential, highways)

- ⚙️ **Self-Contained Training**
  - If the dataset or model does not exist:
    - Data is generated automatically
    - The model is trained automatically
  - No manual preprocessing required

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

**This project does NOT use real-time traffic data.**

- All data is **synthetic**
- Predictions are **not authoritative**
- The project is intended for:
  - Learning
  - Experimentation
  - Demonstration of ML + UI integration

It should **not** be used for navigation, safety, or real-world decision-making.

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

## Images

<img width="1919" height="875" alt="resim" src="https://github.com/user-attachments/assets/9827dcfe-47e7-463d-9e8f-76b460173b95" />
<img width="1919" height="872" alt="resim" src="https://github.com/user-attachments/assets/c8e2fba8-f503-4f25-82b7-37699c85fec7" />

![2026-01-31-12-17-52](https://github.com/user-attachments/assets/2fd1ab54-8be4-4283-8865-feac557ae30f)

## ▶️ Running the App

```bash
pip install streamlit numpy pandas scikit-learn folium streamlit-folium joblib
streamlit run app.py

