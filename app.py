import os
import random
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# -------------------------------------------------------------
# 1. Multi-Dataset Generation
# -------------------------------------------------------------

def generate_hr_data(num_samples=500):
    data = []
    for _ in range(num_samples):
        satisfaction = round(random.uniform(0.1, 1.0), 2)
        tenure = random.randint(1, 10)
        hours = random.randint(120, 320)
        promotion = 1 if random.random() < 0.1 else 0
        
        # Linear logic for churn probability
        logit = (0.5 - satisfaction * 5.0) + (hours / 100.0) + (tenure * 0.2) - (promotion * 2.0)
        prob = 1 / (1 + np.exp(-logit))
        churn = 1 if random.random() < prob else 0
        
        data.append({
            "satisfaction_score": round(satisfaction * 10, 1), 
            "monthly_hours": hours, 
            "years_at_company": tenure, 
            "had_promotion": promotion, 
            "label": churn
        })
    return pd.DataFrame(data)

def generate_marketing_data(num_samples=500):
    data = []
    for _ in range(num_samples):
        has_offer = 1 if random.random() < 0.3 else 0
        has_link = 1 if random.random() < 0.4 else 0
        length = random.randint(10, 300)
        exclamations = random.randint(0, 15)
        
        # Spam logic: high exclamations and short length + offer/link
        logit = -3.0 + (has_offer * 3.0) + (has_link * 2.0) + (exclamations * 0.8) - (length / 50.0)
        prob = 1 / (1 + np.exp(-logit))
        is_spam = 1 if random.random() < prob else 0
        
        data.append({
            "contains_offer": has_offer, 
            "contains_link": has_link, 
            "message_length": length, 
            "exclamation_marks": exclamations, 
            "label": is_spam
        })
    return pd.DataFrame(data)

def generate_loan_data(num_samples=500):
    data = []
    for _ in range(num_samples):
        credit = random.randint(300, 850)
        income = random.randint(20000, 150000)
        loan = random.randint(5000, 50000)
        employment = 1 if random.random() < 0.85 else 0
        
        # Default logic: low credit score and high loan/income ratio
        ratio = loan / income
        logit = (600 - credit) / 100.0 + (ratio * 5.0) - (employment * 3.0)
        prob = 1 / (1 + np.exp(-logit))
        default = 1 if random.random() < prob else 0
        
        data.append({
            "credit_score": credit, 
            "annual_income": income, 
            "loan_amount": loan, 
            "is_employed": employment, 
            "label": default
        })
    return pd.DataFrame(data)

# Master Registry
DATASETS = {
    "hr_churn": {
        "name": "HR: Employee Churn",
        "df": generate_hr_data(),
        "features": ["satisfaction_score", "monthly_hours", "years_at_company", "had_promotion"],
        "label_name": "Churned?", "positive_label": "Left", "negative_label": "Stayed"
    },
    "marketing_spam": {
        "name": "Marketing: Spam Detection",
        "df": generate_marketing_data(),
        "features": ["contains_offer", "contains_link", "message_length", "exclamation_marks"],
        "label_name": "Is Spam?", "positive_label": "Spam", "negative_label": "Genuine"
    },
    "finance_loan": {
        "name": "Finance: Loan Default",
        "df": generate_loan_data(),
        "features": ["credit_score", "annual_income", "loan_amount", "is_employed"],
        "label_name": "Defaulted?", "positive_label": "Default", "negative_label": "Paid back"
    }
}

current_dataset_key = "hr_churn"
current_model = None

# -------------------------------------------------------------
# 2. Endpoints
# -------------------------------------------------------------

@app.route('/', methods=['GET'])
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/config', methods=['GET'])
def get_config():
    """Returns list of available datasets and their metadata."""
    configs = {}
    for k, v in DATASETS.items():
        configs[k] = {
            "name": v["name"],
            "features": v["features"],
            "label_name": v["label_name"],
            "positive": v["positive_label"],
            "negative": v["negative_label"]
        }
    return jsonify(configs)

@app.route('/api/select_dataset', methods=['POST'])
def select_dataset():
    global current_dataset_key, current_model
    key = request.json.get('key')
    if key in DATASETS:
        current_dataset_key = key
        current_model = None # Reset model
        return jsonify({"success": True, "dataset": key})
    return jsonify({"error": "Invalid key"}), 400

@app.route('/api/dataset', methods=['GET'])
def get_dataset():
    ds = DATASETS[current_dataset_key]
    data = ds["df"].to_dict(orient="records")
    return jsonify({"data": data, "config": {"features": ds["features"], "label_name": ds["label_name"]}})

@app.route('/api/train', methods=['POST'])
def train_model():
    global current_model
    ds = DATASETS[current_dataset_key]
    
    selected_features = request.json.get('features', [])
    if not selected_features: return jsonify({"error": "No features"}), 400
    
    X = ds["df"][selected_features].values
    y = ds["df"]['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    current_model = model
    
    y_probs = model.predict_proba(X_test)[:, 1]
    
    importance = [{"feature": f, "weight": float(model.coef_[0][i])} for i, f in enumerate(selected_features)]
    
    # Plot points using first 2 selected features
    x_feat = selected_features[0]
    y_feat = selected_features[1] if len(selected_features) > 1 else selected_features[0]
    x_idx = 0; y_idx = 1 if len(selected_features) > 1 else 0
    
    plot_points = [{"x": float(X_test[i][x_idx]), "y": float(X_test[i][y_idx]), "actual": int(y_test[i]), "prob": float(y_probs[i])} for i in range(len(X_test))]

    return jsonify({
        "importance": importance,
        "test_results": {"y_true": y_test.tolist(), "y_probs": y_probs.tolist()},
        "plot": {"x_label": x_feat, "y_label": y_feat, "points": plot_points},
        "coefficients": {
            "intercept": float(model.intercept_[0]),
            "weights": {f: float(model.coef_[0][i]) for i, f in enumerate(selected_features)}
        }
    })

@app.route('/api/predict', methods=['POST'])
def predict_custom():
    if current_model is None: return jsonify({"error": "Not trained"}), 400
    try:
        data = request.json
        # Convert inputs to float. Backend expects features in order provided during train.
        # Frontend ensures this by sending order.
        X_custom = [float(x) for x in data.get('values', [])]
        probs = current_model.predict_proba([X_custom])[0]
        return jsonify({"probability": float(probs[1])})
    except Exception as e: return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
