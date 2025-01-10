from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Cargar el modelo y el scaler
model = joblib.load('models/Bkm-creditRisk.joblib')
scaler = joblib.load('models/normalizadorScoring.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        # Suponiendo que los datos enviados son una lista de características
        features = np.array(data['features']).reshape(1, -1)
        features_scaled = scaler.transform(features)  # Normaliza los datos
        prediction = model.predict(features_scaled)  # Predicción con el modelo
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '_main_':
    app.run(host='0.0.0.0', port=5000)