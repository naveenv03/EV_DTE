from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
import pickle
import numpy as np
import json

app = Flask(__name__)

# MongoDB Atlas connection
client = MongoClient('mongodb+srv://naveenksv18:naveen@cluster0.xsfvsmk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = client['Range']  # Replace with your database name
collection = db['dte']  # Replace with your collection name

# Load the model
model1 = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        return render_template('index.html')
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index2.html')

    if request.method == 'POST':
        try:
            # Get and convert data from the form
            form_data = {
                'quantity': float(request.form.get('quantity', 0)),
                'City_Roads': float(request.form.get('City_Roads', 0)),
                'Motor_way': float(request.form.get('Motor_way', 0)),
                'country_roads': float(request.form.get('country_roads', 0)),
                'consumption': float(request.form.get('consumption', 0)),
                'A/C': float(request.form.get('A/C', 0)),
                'park_heating': float(request.form.get('park_heating', 0)),
                'avg_speed': float(request.form.get('avg_speed', 0)),
                'ecr_deviation': float(request.form.get('ecr_deviation', 0)),
                'encoded_driving_style': float(request.form.get('encoded_driving_style', 0)),
                'encoded_tire_type': float(request.form.get('encoded_tire_type', 0))
            }

            # Prepare the data for prediction
            input_data = np.array([[form_data['quantity'], form_data['City_Roads'], form_data['Motor_way'],
                                    form_data['country_roads'], form_data['consumption'], form_data['A/C'],
                                    form_data['park_heating'], form_data['avg_speed'], form_data['ecr_deviation'],
                                    form_data['encoded_driving_style'], form_data['encoded_tire_type']]])

            # Predict using the model
            prediction_outcome = model1.predict(input_data)[0]
            
            # Store the result in MongoDB
            collection.insert_one({
                'input_data': form_data,
                'prediction': prediction_outcome
            })

            return render_template('index2.html', p=f"Distance to Empty: {prediction_outcome:.2f} km")

        except Exception as e:
            # Log detailed error message
            print(f"Error occurred: {e}")
            return render_template('index2.html', p="Error occurred during prediction. Check server logs for details.")

@app.route('/visualize', methods=['GET'])
def visualize():
    # Serve the visualization HTML
    return render_template('visualize.html')

@app.route('/api/data', methods=['GET'])
def api_data():
    # Fetch data from MongoDB
    data = list(collection.find({}))

    # Prepare data for the table and charts
    table_data = []
    chart_data = {
        'quantity': [],
        'City_Roads': [],
        'Motor_way': [],
        'country_roads': [],
        'consumption': [],
        'A/C': [],
        'park_heating': [],
        'avg_speed': [],
        'ecr_deviation': [],
        'encoded_driving_style': [],
        'encoded_tire_type': [],
        'prediction': []
    }

    for item in data:
        input_data = item.get('input_data', {})
        prediction = item.get('prediction', 'N/A')

        # Append to table data
        row = {**input_data, 'prediction': prediction}
        table_data.append(row)

        # Append to chart data
        for key in chart_data.keys():
            chart_data[key].append(input_data.get(key, 0))
        chart_data['prediction'].append(prediction)

    return jsonify({'table_data': table_data, 'chart_data': chart_data})

if __name__ == '__main__':
    app.run(debug=True)
