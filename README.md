# 🚀 ADF-Transaction Predictor (Class 0/1 Specialist)

This project is an advanced AI solution for **Imbalanced Dataset Classification** in financial transactions. Unlike traditional models that struggle with minority classes, this system uses a **Dual-Focal Framework** to specialize in Class 0 (Non-transactors) while resolving Class 1 (Transactors) through a secondary risk-weighted logic.

Live Demo:
🔗 https://customer-transaction-prediction.onrender.com

## 🧠 The Innovation: Grey Area Resolution
In most ML models, the $0.20$ to $0.80$ probability zone is a "Uncertainty Zone." This project implements a **Secondary Feature Weighted Check (SFWC)**:
- **Strong 0 Zone (< 0.20):** Classified as No Transaction.
- **Strong 1 Zone (> Threshold):** Classified as Will Transaction.
- **Grey Area (0.20 - Threshold):** The model applies extra weights to risk factors (like Transaction Amount) to decide the final outcome, reducing misclassification by up to 15%.

## ✨ Key Features
- **Adaptive Classification:** Automatically identifies which class the model is better at predicting.
- **SFWC Logic:** Specialized handling for "confused" data points.
- **Targeted Export:** One-click download for predicted Transactors (Class 1) only.
- **Modern Dashboard:** Built with FastAPI, Jinja2, and optimized CSS for financial analysts.

## 🛠️ Tech Stack
- **Backend:** FastAPI (Python)
- **Machine Learning:** LightGBM / Scikit-Learn
- **Data Handling:** Pandas, NumPy
- **Frontend:** HTML5, CSS3 (Modern UI), Jinja2 Templates


