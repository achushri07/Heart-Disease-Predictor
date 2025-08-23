from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# Load model
model = tf.keras.models.load_model("heart_model.keras")

# Load encoders & scaler
label_encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")
y_label = joblib.load("y_label.pkl")   # optional if you want to inverse transform prediction

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect raw input values
    input_data = {
        "Age": int(request.form["Age"]),
        "Sex": request.form["Sex"],
        "RestingBP": int(request.form["RestingBP"]),
        "Cholesterol": int(request.form["Cholesterol"]),
        "FastingBS": request.form["FastingBS"],
        "MaxHR": int(request.form["MaxHR"]),
        "ExerciseAngina": request.form["ExerciseAngina"],
        "Oldpeak": float(request.form["Oldpeak"]),
        "Slope": request.form["Slope"],
        "CA": int(request.form["CA"])
    }

    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # Apply label encoders (same as training)
    for col, le in label_encoders.items():
        df[col] = le.transform(df[col])

    # Scale numeric features
    df_scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(df_scaled)
    pred_class = np.argmax(prediction, axis=1)[0]

    # Map result to readable label
    risk_labels = ["High Risk", "Moderate Risk", "Mild Risk", "No Disease", "Severe Disease"]
    result = risk_labels[pred_class]

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)