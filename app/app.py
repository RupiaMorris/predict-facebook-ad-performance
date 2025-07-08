import streamlit as st
import joblib
import numpy as np
import os

# 🔧 Load the trained model
model_path = os.path.join("..", "models", "random_forest_ctr_model.pkl")
model = joblib.load(model_path)

# 🧾 Page setup
st.set_page_config(page_title="CTR Predictor", layout="centered")
st.title("📊 Facebook Ad CTR Predictor & Budget Planner")

st.markdown("""
Predict the expected **Click-Through Rate (CTR)** for your Facebook ad campaign and estimate how much **budget and impressions** are needed to reach your goal (WhatsApp messages, clicks, etc).
""")

# 🔁 Mappings for encoded inputs
age_range_mapping = {
    "18-24": 1,
    "25-34": 2,
    "35-44": 3,
    "45-54": 4,
    "55-64": 5,
    "65+": 6
}

ad_objective_mapping = {
    "Messages": 1,
    "Clicks": 2,
    "Conversions": 3,
    "Engagement": 4
}

gender_mapping = {
    "Female": 0,
    "Male": 1
}

weekday_mapping = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

# 📥 Sidebar Inputs
st.sidebar.header("Campaign Settings")

target_messages = st.sidebar.number_input("🎯 Target Messages (e.g. WhatsApp chats)", min_value=1, value=10)
age_range = st.sidebar.selectbox("📌 Age Range", list(age_range_mapping.keys()))
gender = st.sidebar.selectbox("🧑 Gender", list(gender_mapping.keys()))
budget = st.sidebar.number_input("💵 Campaign Budget (USD)", min_value=1.0, max_value=10000.0, value=50.0)
ad_objective = st.sidebar.selectbox("🎯 Ad Objective", list(ad_objective_mapping.keys()))
weekday = st.sidebar.selectbox("📅 Day of the Week", list(weekday_mapping.keys()))

# 🔢 Encoded Inputs
encoded_age = age_range_mapping[age_range]
encoded_gender = gender_mapping[gender]
encoded_objective = ad_objective_mapping[ad_objective]
encoded_weekday = weekday_mapping[weekday]

# 📊 Estimations
impressions_per_dollar = 1000
ctr_average = 0.05  # 5%
cost_per_click = 0.20

# Estimated campaign performance
estimated_impressions = impressions_per_dollar * budget
estimated_reach = estimated_impressions * 0.9  # Approximate reach
estimated_clicks = estimated_impressions * ctr_average
results = target_messages if ad_objective == "Messages" else int(estimated_clicks)
cost_per_result = budget / results if results > 0 else 0

# ✅ Aligned Input to Model (order must match training)
input_data = np.array([[encoded_age,            # age
                        encoded_gender,         # gender
                        encoded_objective,      # result_type
                        results,                # results
                        estimated_reach,        # reach
                        estimated_impressions,  # impressions
                        cost_per_result,        # cost_per_result
                        budget,                 # amount_spent_(usd)
                        int(estimated_clicks),  # clicks_(all)
                        encoded_weekday         # weekday
                       ]], dtype=np.float32)

# ✅ Predict Button
if st.button("Predict CTR & Estimate Needs"):
    raw_prediction = model.predict(input_data)[0]
    predicted_ctr = max(0.0, min(raw_prediction, 1.0))  # Clamp to [0, 1]
    predicted_ctr_percent = predicted_ctr * 100

    # Calculate how many impressions and budget would be needed to hit the goal
    if predicted_ctr > 0:
        impressions_needed = target_messages / predicted_ctr
    else:
        impressions_needed = float('inf')

    estimated_budget_needed = impressions_needed / impressions_per_dollar

    # 💬 Output results
    st.success(f"📈 Predicted CTR: **{predicted_ctr_percent:.2f}%**")
    st.info(f"📌 Estimated impressions needed for {target_messages} messages: **{impressions_needed:,.0f}**")
    st.info(f"💰 Estimated budget required: **${estimated_budget_needed:.2f} USD**")
