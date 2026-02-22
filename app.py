from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained model
model_path = os.path.join(os.path.dirname(__file__), 'regmodel.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        features = [
            float(data['MedInc']),
            float(data['HouseAge']),
            float(data['AveRooms']),
            float(data['AveBedrms']),
            float(data['Population']),
            float(data['AveOccup']),
            float(data['Latitude']),
            float(data['Longitude']),
        ]

        input_array = np.array(features).reshape(1, -1)
        prediction = model.predict(input_array)[0]

        # The model output is in units of $100,000
        price_in_dollars = round(prediction * 100000, 2)

        return jsonify({
            'success': True,
            'prediction': price_in_dollars,
            'prediction_formatted': f"${price_in_dollars:,.0f}"
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
