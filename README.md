# Heart Disease Risk Predictor

This project is a **Deep Learning based web application** that predicts the risk of **Heart Disease** using an **Artificial Neural Network (ANN)**.  
It uses **Flask** as the backend, along with **HTML & CSS** for the frontend interface.

---

## Features
- Built using **TensorFlow/Keras** (ANN model with multiclass classification).
- Preprocessing with **Label Encoding** and **StandardScaler**.
- Model training with **Early Stopping** to avoid overfitting.
- Interactive web app with Flask + HTML + CSS.
- Modal popup to show predicted condition in a user-friendly format.
- Five possible risk categories:
  - High Risk  
  - Moderate Risk  
  - Mild Risk  
  - No Disease  
  - Severe Disease  

---

## Screenshot
<img width="1347" height="631" alt="Screenshot 2025-08-23 113017" src="https://github.com/user-attachments/assets/d2be3222-50d3-4b67-9023-c7792009b4a6" />

---

## Model Training (Summary)

- Dataset: heart_disease_dataset_multiclass.csv
- Preprocessing:
  - Label encoding for categorical features (Sex, FastingBS, ExerciseAngina, Slope) 
  - StandardScaler for numerical features 
- Model:
  - Input Layer → Hidden Layer (10 neurons, ReLU) → Output Layer (5 neurons, Softmax)
  - Optimizer: Adam (lr=0.01)
  - Loss: Sparse Categorical Crossentropy
  - Callback: EarlyStopping (patience=20)
Performance plots during training:
- Accuracy vs Epochs
- Loss vs Epochs

---

## Tech Stack
- **Python**  
- **Flask**  
- **TensorFlow / Keras**  
- **scikit-learn**  
- **pandas, numpy**  
- **HTML, CSS**  

---

## Files Included
- `README.md` – This file
- `heart_disease_dataset_multiclass` – The dataset used
- `model.ipynb` – Python Notebook for model training
- `app.py` – Python File for flask
- `templates/index.html` – HTML file
- `static/style.css` – CSS file
- `requirements.txt` – Text file containing required libraries to install

---

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you’d like to change.
