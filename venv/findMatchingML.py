import os
import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error  # Added import statement for mean_squared_error


# Constants
MODEL_FILENAME = 'trained_model.h5'
SCALER_FILENAME = 'scaler.pkl'
FILEPATH = 'pair.json'  # Replace with the actual file path

# Function to load and preprocess the data
def load_and_preprocess_data(filepath):
    data = pd.read_json(filepath)
    y_values = data['y'].apply(lambda y: [int(x.split('-')[1]) for x in y.split('::')])
    y_matrix = np.array(y_values.tolist())
    scaler = MinMaxScaler(feature_range=(0, 1))
    y_matrix_normalized = scaler.fit_transform(y_matrix)
    return y_matrix_normalized, scaler

# Function to create sequences from data
def create_sequences(data, sequence_length=5):
    x, y = [], []
    for i in range(len(data) - sequence_length):
        x.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(x), np.array(y)

# Function to build the LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(units=input_shape[1]))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def format_new_data(new_y_value, scaler, sequence_length=5):
    new_data = [int(x.split('-')[1]) for x in new_y_value.split('::')]
    # Create a dummy sequence with the new data repeated
    dummy_sequence = [new_data] * sequence_length
    normalized_sequence = scaler.transform(dummy_sequence)
    return normalized_sequence.reshape(1, sequence_length, -1)

def find_closest_match(predicted_y, original_data, scaler):
    # Inverse transform the predicted 'y' value
    inv_predicted_y = scaler.inverse_transform(predicted_y)

    # Initialize variables to store the minimum error and corresponding 'y' value
    min_error = float('inf')
    closest_match = None

    # Loop through each row in the original data to find the closest match
    for row in original_data:
        # Transform the row for error calculation
        transformed_row = scaler.transform([row])
        # Calculate mean squared error
        error = mean_squared_error(inv_predicted_y, transformed_row)
        if error < min_error:
            min_error = error
            closest_match = row

    return closest_match, min_error

# Main function
def main():
    # Load and preprocess data
    data, scaler = load_and_preprocess_data(FILEPATH)
    
    # Prepare original_data for comparison
    y_values = data['y'].apply(lambda y: [int(x.split('-')[1]) for x in y.split('::')])
    original_data = np.array(y_values.tolist())

    if os.path.exists(MODEL_FILENAME) and os.path.exists(SCALER_FILENAME):
        # Load the trained model and scaler
        model = load_model(MODEL_FILENAME)
        scaler = pd.read_pickle(SCALER_FILENAME)
    else:
        # Create sequences from data
        x, y = create_sequences(data, sequence_length=5)

        # Build and train the model
        model = build_model(x.shape[1:])
        model.fit(x, y, epochs=100, batch_size=32)

        # Save the model and the scaler
        model.save(MODEL_FILENAME)
        pd.to_pickle(scaler, SCALER_FILENAME)

    # New 'y' value for prediction
    new_y_value = '1-1::2-0::3-24::4-25::5-22::6-8'

    # Format and predict
    formatted_data = format_new_data(new_y_value, scaler, sequence_length=5)
    predicted_y = model.predict(formatted_data)

    # Find the closest match in the original dataset
    closest_match, error = find_closest_match(predicted_y, original_data, scaler)
    print('Closest match \'y\' value:', closest_match)
    print('Error (Mean Squared Error) of match:', error)

    # Inverse transform to get the actual predicted 'y' value
    predicted_y_actual = scaler.inverse_transform(predicted_y)
    print('Predicted next \'y\' value:', predicted_y_actual)

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
