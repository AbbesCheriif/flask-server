import pickle
from flask import Flask, request, jsonify
import numpy as np
from pymongo import MongoClient

# Initialize Flask app
app = Flask(__name__)

# Load the models
with open('models/scaler_new.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/pca_new.pkl', 'rb') as f:
    pca = pickle.load(f)

with open('models/label_encoder_new.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('models/kmeans_new.pkl', 'rb') as f:
    kmeans = pickle.load(f)

client = MongoClient("mongodb+srv://authentificationmdp:AminePFE888@cluster0.8tcz9.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["PFE"]
collection = db["buildings"]

@app.route('/')
def home():
    return "Welcome to the ML Model API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        data = request.json
        features = np.array(data['features']).reshape(1, -1)

        # Preprocess features
        scaled_features = scaler.transform(features)
        reduced_features = pca.transform(scaled_features)

        # Predict with KMeans
        cluster = kmeans.predict(reduced_features)

        # Decode label (if applicable)
        cluster_label = label_encoder.inverse_transform(cluster)

        return jsonify({'cluster': cluster_label[0]})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/data', methods=['GET'])
def get_data():
    data = collection.find()
    result = []
    for document in data:
        document['_id'] = str(document['_id'])  # Convert ObjectId to string
        result.append(document)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
