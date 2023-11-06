import pickle
import xgboost as xgb
from flask import Flask
from flask import request, jsonify

def load(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

dv, model = load("model_xgb.bin")

app = Flask("cardio")

@app.route("/predict", methods=["POST"])
def predict():
    patient = request.get_json()
    X = dv.transform([patient])
    dX = xgb.DMatrix(X, feature_names=dv.get_feature_names_out().tolist())
    y_pred = model.predict(dX)
    decision = y_pred >= 0.5

    result = {
        "cardio_probability": float(y_pred),
        "has_disease": bool(decision)
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)