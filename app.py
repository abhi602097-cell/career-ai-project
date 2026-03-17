from flask import Flask, request, render_template, redirect, session
import pandas as pd
import numpy as np
import joblib
import os

# 🔹 Optional GPT (safe import)
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
app.secret_key = "supersecretkey123"  # change in production

# 🔹 Load ML files
model = joblib.load("career_model.pkl")
selector = joblib.load("feature_selector.pkl")
label_encoder = joblib.load("label_encoder.pkl")
model_columns = joblib.load("model_columns.pkl")


# =========================
# 🔥 GPT EXPLANATION
# =========================
def generate_gpt_explanation(input_data, prediction):

    if client is None:
        return f"This career ({prediction}) fits your profile based on your academic and aptitude performance."

    prompt = f"""
You are a career counselor.

User data:
{input_data}

Predicted career: {prediction}

Explain:
1. Why this fits
2. Strengths
3. Improvements

Keep it simple and motivational.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        return response.choices[0].message.content

    except Exception as e:
        print("GPT ERROR:", e)
        return f"This career ({prediction}) aligns with your profile."


# =========================
# 🔥 ROADMAP GENERATOR
# =========================
def generate_roadmap(prediction):

    roadmap_map = {
        "data": ["Learn Python", "Statistics", "Machine Learning", "Projects"],
        "management": ["Communication", "Leadership", "MBA / Strategy"],
        "sales": ["Negotiation", "Customer Psychology", "CRM Tools"],
        "engineering": ["Core Concepts", "Projects", "Internships"],
        "trades": ["Technical Skills", "Apprenticeship", "Hands-on Work"]
    }

    for key in roadmap_map:
        if key in prediction.lower():
            return roadmap_map[key]

    return ["Skill Development", "Domain Knowledge", "Experience"]


# =========================
# 🔥 INSIGHTS GENERATOR
# =========================
def generate_insight(prediction):

    if "data" in prediction.lower():
        return {"salary": "6-20 LPA", "growth": "Very High"}

    elif "manager" in prediction.lower():
        return {"salary": "8-25 LPA", "growth": "High"}

    elif "sales" in prediction.lower():
        return {"salary": "3-12 LPA", "growth": "High"}

    elif "engineer" in prediction.lower():
        return {"salary": "4-15 LPA", "growth": "High"}

    return {"salary": "2-8 LPA", "growth": "Moderate"}


# =========================
# 🏠 ROUTES
# =========================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = request.form.to_dict()

        # 🔹 Rename fields
        rename_map = {
            "ASVAB_MATH_VERBAL_SCORE": "ASVAB_MATH_VERBAL_SCORE_PCT_XRND",
            "CV_BA_CREDITS": "CV_BA_CREDITS_01_2002",
            "EDUCATION": "CV_HIGHEST_DEGREE_0809_2008",
            "COLLEGE_STATUS": "SCH_COLLEGE_STATUS_2002.01",
            "REGION": "CV_CENSUS_REGION_1997"
        }

        for old, new in rename_map.items():
            if old in form_data:
                form_data[new] = form_data.pop(old)

        # 🔹 Create DataFrame
        df = pd.DataFrame([form_data])

        # 🔹 Convert numeric fields
        numeric_cols = [
            "ASVAB_MATH_VERBAL_SCORE_PCT_XRND",
            "CV_BA_CREDITS_01_2002"
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df[numeric_cols] = df[numeric_cols].fillna(0)

        # 🔹 Fill missing
        df.fillna("Unknown", inplace=True)

        # 🔹 One-hot encoding
        df = pd.get_dummies(df)

        # 🔹 Match training columns
        for col in model_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[model_columns]

        # 🔹 Feature selection
        X = selector.transform(df)

        # 🔹 Prediction
        probs = model.predict_proba(X)[0]
        top3_idx = np.argsort(probs)[-3:][::-1]

        predictions = []
        for i in top3_idx:
            predictions.append({
                "career": label_encoder.inverse_transform([i])[0],
                "probability": round(float(probs[i]) * 100, 2)
            })

        top_prediction = predictions[0]["career"]

        # 🔥 AI + Dynamic Layers
        explanation = generate_gpt_explanation(form_data, top_prediction)
        roadmap = generate_roadmap(top_prediction)
        insight = generate_insight(top_prediction)

        # 🔹 Store in session
        session["result"] = {
            "predictions": predictions,
            "top_prediction": predictions[0],
            "explanation": explanation,
            "roadmap": roadmap,
            "insight": insight
        }

        return redirect("/dashboard")

    except Exception as e:
        print("ERROR:", e)
        return f"Error occurred: {str(e)}"


@app.route("/dashboard")
def dashboard():
    data = session.get("result")
    if not data:
        return redirect("/")
    return render_template("dashboard.html", data=data)


@app.route("/explanation")
def explanation_page():
    data = session.get("result")
    if not data:
        return redirect("/")
    return render_template("explanation.html", data=data)


# =========================
# 🚀 RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(debug=True)
