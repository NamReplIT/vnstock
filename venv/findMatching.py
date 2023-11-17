import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense,Dropout,Flatten,Bidirectional,BatchNormalization,ReLU,Conv1D,MaxPooling1D
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.metrics import Precision, Recall
from keras.regularizers import L1L2,l1_l2,l2
from keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split
import os,json
from input import inputs
from keras.callbacks import CSVLogger

json_path = './mostRepeated.json'

numOfSet = 3

numOfFeature = 2

sliceStart = 0

sliceEnd = 1



# Load and preprocess data
def load_data(filepath):
    data = pd.read_json(filepath)
    return data

def load_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def load_data(filepath):
    data = pd.read_json(filepath)
    return data

def preprocess_y_values(y_values):
    processed = []
    for y in y_values:
        components = [int(part.split('-')[1]) for part in y.split('::')]
        processed.append(components)
    return processed

def convertIndex(items):

    with open(json_path, 'r') as file:
        json_dataset = json.load(file)

    list_result = []

    for item in items:

        results = []
        for key in item:
            try:
                index = json_dataset[key].index(item[key])
            except ValueError:
                index = 'N/A'
            results.append(f"{key.split('_')[1]}-{index}")

        list_result.append('::'.join(results))

    return list_result

def preprocess_input(y_values):
    return np.array(preprocess_y_values(y_values))

def get_specific_columns(data, start_index, end_index):

    if start_index > end_index:
        raise ValueError("Start index cannot be greater than end index.")

    # Extract and return the specific columns (2nd to 4th) from the specified rows
    return [row[start_index:end_index+1] for row in data]

def create_sequences(data, sequence_length=3):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i+sequence_length])
        targets.append(data[i+sequence_length])
    return np.array(sequences), np.array(targets)


# Postprocess model predictions
def postprocess_prediction(predicted_y):
    return np.round(predicted_y).astype(int)

def get_values_from_indices(indices):
    json_data = load_json('./mostRepeated.json')
    result = []
    for i, index in enumerate(indices[0]):  # Assuming prediction_indices is nested
        # Correct for zero-based indexing
        num_key = 'num_{:02d}'.format(i + 1)
        if num_key in json_data and 0 <= index < len(json_data[num_key]):
            result.append(json_data[num_key][index])
        else:
            result.append(None)  # Append None if the key doesn't exist or index is out of range
    return result

def build_model(input_shape, numOfFeature=3):

    model = Sequential([
        LSTM(8,input_shape=input_shape),
        Dropout(0.1),
        Dense(numOfFeature, activation='swish')
    ])

    # Compile the model using Adam optimizer and categorical cross-entropy loss
    model.compile(optimizer=SGD(momentum=0.5), loss='mse', metrics=['accuracy', Precision(), Recall()])
 
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    return model, early_stopping

# Main function
def main():
    model_file = 'final_model.h5'

    filepath = './pair.json'  # Update with your file path

    data = load_data(filepath)

    processed_y = preprocess_y_values(data['y'])

    processed_y = get_specific_columns(processed_y, sliceStart, sliceEnd)
    
    # X, y = create_sequences(processed_y, sequence_length)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Check if model exists
    if os.path.exists(model_file):
        # Load existing model
        model = load_model(model_file)
    else:
        csv_logger = CSVLogger('training_log.csv', append=True, separator=';')

        X, y = create_sequences(processed_y, numOfSet)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

        model,early_stopping = build_model(X_train.shape[1:], numOfFeature)

        checkpoint = ModelCheckpoint('model.h5', save_best_only=True)

        model.fit(X_train, y_train, epochs=500,batch_size=1, validation_data=(X_test, y_test), callbacks=[checkpoint,early_stopping,csv_logger])

        # Save the final model
        #model.save(model_file)

    # Use model for prediction
    # Example input y value for prediction

    processed_input_y = preprocess_y_values(convertIndex(inputs))

    processed_input_y = get_specific_columns(processed_input_y, sliceStart, sliceEnd)

    processed_input_y = np.array(processed_input_y)

    processed_input_y = processed_input_y.reshape((-1, numOfSet, numOfFeature))

    print("processed_input_y:", processed_input_y)

    predicted_y = model.predict(processed_input_y)

    postprocessed_prediction = postprocess_prediction(predicted_y)

    print("Predicted next y value:", postprocessed_prediction)

    print("value ", get_values_from_indices(postprocessed_prediction))

if __name__ == "__main__":
    main()
