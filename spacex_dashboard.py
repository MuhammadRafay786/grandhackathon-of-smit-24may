# 📁 SpaceX Launch Success Prediction Dashboard (All-in-One Code)
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 🚀 Step 1: Data Load karo (CSV file se)
import os

csv_path = r"c:\Users\AI web0.3\OneDrive\Desktop\AI and datascence hacakthon\spacex_cleaned.csv"
if not os.path.exists(csv_path):
    st.error(f"CSV file not found at: {csv_path}\n\nPlease check the file path and make sure the file exists.")
    st.stop()
df = pd.read_csv(csv_path)  # Update the path if your CSV is located elsewhere

# 🧹 Step 2: Data Cleaning & NaN handling
df = df.dropna(subset=['success'])  # success column me NaN to hatao
df['payload_mass'] = df['payload_mass'].fillna(df['payload_mass'].mean())
df['orbit'] = df['orbit'].fillna(df['orbit'].mode()[0])

# Ensure 'success' is binary (0/1) and integer type
if df['success'].dtype != int and df['success'].dtype != 'int64':
    # Try to map common string values to 0/1
    if set(df['success'].unique()) <= set(['Success', 'Fail', 'success', 'fail', 'SUCCESS', 'FAIL']):
        df['success'] = df['success'].str.lower().map({'success': 1, 'fail': 0})
    else:
        df['success'] = pd.to_numeric(df['success'], errors='coerce')
df['success'] = df['success'].astype(int)

# 🔠 Step 3: Encode Categorical Columns
le_orbit = LabelEncoder()
le_site = LabelEncoder()
le_rocket = LabelEncoder()

df['orbit'] = le_orbit.fit_transform(df['orbit'].astype(str))
df['site_name'] = le_site.fit_transform(df['site_name'].astype(str))
df['rocket_name'] = le_rocket.fit_transform(df['rocket_name'].astype(str))

# 🎯 Step 4: Features (X) aur Target (y) define karo
X = df[['payload_mass', 'orbit', 'site_name', 'rocket_name']]
y = df['success']

# 🔀 Step 5: Train/Test Split (Optional testing ke liye)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🌲 Step 6: Random Forest Model Train karo
model = RandomForestClassifier(class_weight='balanced', random_state=42, min_samples_leaf=1, max_features='sqrt')
model.fit(X_train, y_train)

# 🧠 Step 7: Streamlit App UI Start
st.set_page_config(page_title="SpaceX Launch Predictor", layout="centered")
st.title("🚀 SpaceX Launch Success Predictor")
st.markdown("Predict karo ke koi launch successful hoga ya fail based on historical data.")

# 🎛 User Inputs
payload = st.slider("Payload Mass (kg)", 0, 10000, 5000)
orbit_input = st.selectbox("Orbit Type", le_orbit.classes_)
site_input = st.selectbox("Launch Site", le_site.classes_)
rocket_input = st.selectbox("Rocket Name", le_rocket.classes_)

# 🧾 Step 8: Encode user inputs
user_input_df = pd.DataFrame({
    'payload_mass': [payload],
    'orbit': [le_orbit.transform([orbit_input])[0]],
    'site_name': [le_site.transform([site_input])[0]],
    'rocket_name': [le_rocket.transform([rocket_input])[0]]
})

# 🔮 Step 9: Prediction button
target = model.predict(user_input_df)[0]

if st.button("Predict"):
    if target == 1:
        st.success("✅ Launch is likely to be SUCCESSFUL!")
    else:
        st.error("❌ Launch may FAIL.")

# 📊 Optional: Accuracy Report (developer view only)
with st.expander("See model evaluation report"):
    y_pred = model.predict(X_test)
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    st.text("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))