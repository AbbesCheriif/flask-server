import joblib
import json
from flask import Flask, jsonify
import numpy as np
from pymongo import MongoClient
from bson.objectid import ObjectId
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# Initialize Flask app
app = Flask(__name__)

# Load the models
pca = joblib.load('models/pca_new.pkl')
if not isinstance(pca, PCA):
    raise TypeError(f"Expected PCA, got {type(pca)}")

scaler = joblib.load('models/scaler_new.pkl')
if not isinstance(scaler, MinMaxScaler):
    raise TypeError(f"Expected MinMaxScaler, got {type(scaler)}")

label_encoder_city = joblib.load('models/label_encoder_city_new.pkl')
if not isinstance(label_encoder_city, LabelEncoder):
    raise TypeError(f"Expected LabelEncoder, got {type(label_encoder)}")

label_encoder_type = joblib.load('models/label_encoder_type_new.pkl')
if not isinstance(label_encoder_type, LabelEncoder):
    raise TypeError(f"Expected LabelEncoder, got {type(label_encoder)}")

kmeans = joblib.load('models/kmeans_new.pkl')
# if not isinstance(kmeans, KMeans):
#     raise TypeError(f"Expected KMeans, got {type(kmeans)}")

# Load city mapping from JSON file
with open('cities_out.json', 'r', encoding='utf-8') as f:
    city_mapping = json.load(f)

# Connect to MongoDB Atlas
client = MongoClient(
    "mongodb+srv://authentificationmdp:AminePFE888@cluster0.8tcz9.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)
db = client["PFE"]
collection = db["buildings"]
collection_user = db["RecommendationForm"]

@app.route('/')
def home():
    return "Welcome to the ML Model API!"

def serialize_document(document):
    """Recursively serialize MongoDB document."""
    if isinstance(document, ObjectId):
        return str(document)
    elif isinstance(document, dict):
        return {key: serialize_document(value) for key, value in document.items()}
    elif isinstance(document, list):
        return [serialize_document(item) for item in document]
    return document  # Return the value as is if it's already serializable

@app.route('/data', methods=['GET'])
def get_data():
    try:
        data = collection.find()
        result = [serialize_document(document) for document in data]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/update-building-clusters', methods=['POST'])
def update_clusters():
    try:
        # Fetch all documents from the collection
        buildings = list(collection.find())

        for building in buildings:
            # Extract features from the building document
            city = building.get('city', '').strip()
            nb_rooms = building.get('rooms', '')
            building_type = building.get('type', '').strip()
            area = building.get('area', '')
            price = building.get('price', '')

            # Map small town to main city
            main_city = None
            for big_city, towns in city_mapping.items():
                if city in towns:
                    main_city = big_city
                    break

            # Skip if no valid main city found
            if not main_city:
                continue

            

            # Create feature array
            features = [nb_rooms, price, area, main_city, building_type]

            # Create a DataFrame for the features
            feature_columns = ['BHK', 'Rent', 'Size', 'City', 'Type']
            features_df = pd.DataFrame([features], columns=feature_columns)

            type_mapping = {
                'appartement': 'apartment',  # Replace with standard label
                'flat': 'flat'              # Add other mappings if needed
            }

            features_df['Type'] = features_df['Type'].replace(type_mapping)
            print(features_df)
            # Handle unseen categories for LabelEncoder
            # Transform 'main_city' and 'building_type' using the LabelEncoder
            features_df['City'] = label_encoder_city.transform(features_df['City'])
            features_df['Type'] = label_encoder_type.transform(features_df['Type'])
            print(features_df)

            columns_to_scale = ['City','BHK','Rent','Size','Type']
            # Apply scaling to the features
            features_df[columns_to_scale] = scaler.transform(features_df[columns_to_scale])
            print('hello')

            # Apply PCA transformation to the scaled features
            reduced_features = pca.transform(features_df)
            

            # Predict the cluster using the KMeans model
            cluster = kmeans.predict(reduced_features)[0]

            # Update the document in the MongoDB collection with the predicted cluster
            collection.update_one(
                {'_id': building['_id']},
                {'$set': {'cluster': int(cluster)}}
            )

        return jsonify({'status': 'success', 'message': 'Clusters updated successfully!'})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})




@app.route('/update-user-clusters', methods=['POST'])
def update_user_clusters():
    try:
        # Fetch all documents from the collection
        users = list(collection_user.find())

        for user in users:
            # Extract features from the building document
            city = user.get('city', '').strip()
            nb_rooms = user.get('numberOfRooms', '')
            building_type = user.get('type', '').strip()
            area = user.get('area', '')
            price = user.get('price', '')

            # Map small town to main city
            main_city = None
            for big_city, towns in city_mapping.items():
                if city in towns:
                    main_city = big_city
                    break

            # Skip if no valid main city found
            if not main_city:
                continue

            

            # Create feature array
            features = [nb_rooms, price, area, main_city, building_type]

            # Create a DataFrame for the features
            feature_columns = ['BHK', 'Rent', 'Size', 'City', 'Type']
            features_df = pd.DataFrame([features], columns=feature_columns)

            type_mapping = {
                'appartement': 'apartment',  # Replace with standard label
                'flat': 'flat'              # Add other mappings if needed
            }

            features_df['Type'] = features_df['Type'].replace(type_mapping)
            print(features_df)
            # Handle unseen categories for LabelEncoder
            # Transform 'main_city' and 'building_type' using the LabelEncoder
            features_df['City'] = label_encoder_city.transform(features_df['City'])
            features_df['Type'] = label_encoder_type.transform(features_df['Type'])
            print(features_df)

            columns_to_scale = ['City','BHK','Rent','Size','Type']
            # Apply scaling to the features
            features_df[columns_to_scale] = scaler.transform(features_df[columns_to_scale])
            print('hello')

            # Apply PCA transformation to the scaled features
            reduced_features = pca.transform(features_df)
            

            # Predict the cluster using the KMeans model
            cluster = kmeans.predict(reduced_features)[0]

            # Update the document in the MongoDB collection with the predicted cluster
            collection_user.update_one(
                {'_id': user['_id']},
                {'$set': {'cluster': int(cluster)}}
            )

        return jsonify({'status': 'success', 'message': 'Clusters updated successfully!'})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
