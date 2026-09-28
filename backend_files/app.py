
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import io

app = Flask(__name__)

# Load the serialized model pipeline
model = joblib.load("superkart_model.joblib")

# Expected input features
FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category"
]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "SuperKart Sales Prediction API is running."
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Request body is empty."}), 400

        # Accept either a single record or a list of records
        if isinstance(data, dict):
            input_data = [data]
        elif isinstance(data, list):
            input_data = data
        else:
            return jsonify({
                "error": "Request body must be a JSON object or a list of objects."
            }), 400

        df_input = pd.DataFrame(input_data)

        # Check for missing required columns
        missing_columns = [col for col in FEATURES if col not in df_input.columns]

        if missing_columns:
            return jsonify({
                "error": "Missing required columns.",
                "missing_columns": missing_columns
            }), 400

        # Keep only the features used by the model, in the correct order
        df_input = df_input[FEATURES]

        # Generate predictions
        predictions = model.predict(df_input)

        return jsonify({
            "predictions": predictions.tolist()
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    try:
        if "file" not in request.files:
            return jsonify({
                "error": "No CSV file was provided."
            }), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({
                "error": "No file selected."
            }), 400

        # Read uploaded CSV
        df_input = pd.read_csv(file)

        # Check for missing required columns
        missing_columns = [col for col in FEATURES if col not in df_input.columns]

        if missing_columns:
            return jsonify({
                "error": "Missing required columns.",
                "missing_columns": missing_columns
            }), 400

        # Keep model input columns in the correct order
        df_model_input = df_input[FEATURES]

        # Generate predictions
        predictions = model.predict(df_model_input)

        # Add predictions to the original data
        result = df_input.copy()
        result["Predicted_Product_Store_Sales_Total"] = predictions

        return jsonify(
            result.to_dict(orient="records")
        )

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
