import streamlit as st
import pandas as pd
import numpy as np
import os
import sqlite3
import hashlib
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Database Setup
conn = sqlite3.connect("loan_defaults.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")
conn.commit()

# User Authentication
def register_user(username, password):
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
    conn.commit()

def authenticate_user(username, password):
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hashed_password))
    return cursor.fetchone()

# Streamlit UI
st.title("🏦 Loan Default Prediction using Machine Learning & Deep Neural Networks")

menu = st.sidebar.selectbox("Menu", ["Register", "Login", "Upload & Train Model", "Predict Loan Default"])

# Registration
if menu == "Register":
    st.subheader("🔑 Register")
    reg_username = st.text_input("Choose a Username")
    reg_password = st.text_input("Choose a Password", type="password")
    
    if st.button("Register"):
        try:
            register_user(reg_username, reg_password)
            st.success("✅ Registration Successful. You can now log in.")
        except:
            st.error("❌ Username already exists. Try another.")

# Login
if menu == "Login":
    st.subheader("🔐 Login")
    login_username = st.text_input("Username")
    login_password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if authenticate_user(login_username, login_password):
            st.session_state["logged_in"] = True
            st.session_state["username"] = login_username
            st.success(f"✅ Welcome, {login_username}!")
        else:
            st.error("❌ Invalid credentials.")

# Upload & Train Model
if menu == "Upload & Train Model" and "logged_in" in st.session_state:
    st.subheader("📂 Upload Loan Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📊 Preview of Dataset", df.head())

        if st.button("Train Model"):
            # Preprocessing
            st.write("🔄 Processing Data...")
            df = df.dropna()  # Remove missing values
            X = df.iloc[:, :-1]  # Features
            y = df.iloc[:, -1]   # Target (Loan Default)

            # Encode categorical labels
            if y.dtype == 'object':
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)

            # Split dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Train Random Forest (Baseline ML Model)
            st.write("🌲 Training Random Forest Model...")
            rf_model = RandomForestClassifier(n_estimators=100)
            rf_model.fit(X_train, y_train)
            rf_preds = rf_model.predict(X_test)
            rf_accuracy = accuracy_score(y_test, rf_preds)
            st.write(f"✅ Random Forest Accuracy: {rf_accuracy:.2%}")

            # Train Deep Neural Network (DNN)
            st.write("🧠 Training Deep Neural Network (DNN)...")
            model = Sequential([
                Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
                Dropout(0.2),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

            dnn_loss, dnn_accuracy = model.evaluate(X_test, y_test)
            st.write(f"✅ DNN Accuracy: {dnn_accuracy:.2%}")

            # Save models
            joblib.dump(rf_model, "random_forest_model.pkl")
            model.save("dnn_model.h5")
            st.success("📂 Models Saved Successfully!")

# Predict Loan Default
if menu == "Predict Loan Default" and "logged_in" in st.session_state:
    st.subheader("📊 Predict Loan Default")
    
    # Load trained models
    if os.path.exists("random_forest_model.pkl") and os.path.exists("dnn_model.h5"):
        rf_model = joblib.load("random_forest_model.pkl")
        dnn_model = keras.models.load_model("dnn_model.h5")
    else:
        st.error("❌ No trained model found. Train a model first!")
        st.stop()

    # User input form
    st.write("🔢 Enter Borrower Data:")
    age = st.number_input("Age", 18, 100)
    income = st.number_input("Annual Income ($)", 1000, 1000000)
    credit_score = st.number_input("Credit Score", 300, 850)
    loan_amount = st.number_input("Loan Amount ($)", 1000, 500000)
    loan_term = st.number_input("Loan Term (Months)", 12, 360)
    interest_rate = st.number_input("Interest Rate (%)", 0.1, 20.0)
    num_defaults = st.number_input("Past Defaults", 0, 10)

    input_data = np.array([[age, income, credit_score, loan_amount, loan_term, interest_rate, num_defaults]])

    # Standardize input (assuming scaler from training)
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)

    if st.button("Predict Loan Default"):
        rf_prediction = rf_model.predict(input_data_scaled)
        dnn_prediction = dnn_model.predict(input_data_scaled)
        dnn_prediction = (dnn_prediction > 0.5).astype(int)

        st.write(f"📌 **Random Forest Prediction:** {'Default' if rf_prediction[0] == 1 else 'No Default'}")
        st.write(f"🧠 **DNN Prediction:** {'Default' if dnn_prediction[0][0] == 1 else 'No Default'}")

        if rf_prediction[0] == 1 or dnn_prediction[0][0] == 1:
            st.error("⚠️ High Risk: Loan Default Likely!")
        else:
            st.success("✅ Low Risk: Loan Repayment Likely!")

