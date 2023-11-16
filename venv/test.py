import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense,Dropout,Flatten,Bidirectional,BatchNormalization,ReLU,Conv1D,MaxPooling1D
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.metrics import Precision, Recall
from keras.regularizers import L1L2,l1_l2,l2
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import os,json
from convertIndex import convertIndex
from keras.callbacks import CSVLogger


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

def get_specific_columns(data, start_index, end_index):
    """
    Extracts columns 2 to 4 (indices 1 to 3) from rows in the data starting at 'start_index' and ending at 'end_index' (inclusive).
    
    :param data: The input data, a list of lists.
    :param start_index: The starting index of the rows to extract.
    :param end_index: The ending index of the rows to extract.
    :return: A list of lists containing the specified columns from the specified rows.
    """
    # Ensure that start and end indices are within the bounds of the data
    if start_index < 0 or end_index >= len(data):
        raise ValueError("Start or end index is out of range.")
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

# Preprocess input for prediction
def preprocess_input_y(y_values):

    list_result = []

    for y_value in y_values:
        components = [int(part.split('-')[1]) for part in y_value.split('::')]
        list_result.append(components)
    
    return np.array([list_result])


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
            print(num_key,index)
            result.append(None)  # Append None if the key doesn't exist or index is out of range
    return result

def build_model(input_shape):

    model = Sequential([
        LSTM(720, activation='tanh', input_shape=input_shape,
             kernel_regularizer=l1_l2(l1=0.01/5, l2=0.01/5)),
        Dropout(0.8),
        
        Dense(6, activation='relu')
    ])

    # Compile the model using Adam optimizer and categorical cross-entropy loss
    model.compile(optimizer=Adam(learning_rate=0.002), loss='categorical_crossentropy', 
                  metrics=['accuracy', Precision(), Recall()])
 
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    return model, early_stopping

# Main function
def main():

    # Use model for prediction
    # Example input y value for prediction
    # input_y = convertIndex({
    #     "num_01": "01",
    #     "num_02": "04",
    #     "num_03": "10",
    #     "num_04": "13",
    #     "num_05": "14",
    #     "num_06": "44"
    # })
    # input_y = convertIndex({
    #     "num_01": "07",
    #     "num_02": "20",
    #     "num_03": "23",
    #     "num_04": "27",
    #     "num_05": "31",
    #     "num_06": "33"
    # })
    # input_y = convertIndex({
    #     "num_01": "01",
    #     "num_02": "13",
    #     "num_03": "16",
    #     "num_04": "18",
    #     "num_05": "23",
    #     "num_06": "25"
    # })
    # input_y = convertIndex({
    #     "num_01": "02",
    #     "num_02": "07",
    #     "num_03": "09",
    #     "num_04": "13",
    #     "num_05": "22",
    #     "num_06": "38"
    # })
    # input_y = convertIndex({
    #     "num_01": "01",
    #     "num_02": "03",
    #     "num_03": "15",
    #     "num_04": "16",
    #     "num_05": "23",
    #     "num_06": "28"
    # })
    # input_y = convertIndex({
    #     "num_01": "05",
    #     "num_02": "07",
    #     "num_03": "15",
    #     "num_04": "21",
    #     "num_05": "32",
    #     "num_06": "45"
    # })
    # input_y = convertIndex({
    #     "num_01": "07",
    #     "num_02": "10",
    #     "num_03": "14",
    #     "num_04": "21",
    #     "num_05": "26",
    #     "num_06": "37"
    # })
    # input_y = convertIndex({
    #     "num_01": "04",
    #     "num_02": "06",
    #     "num_03": "13",
    #     "num_04": "25",
    #     "num_05": "31",
    #     "num_06": "41"
    # })
    # input_y = convertIndex({
    #     "num_01": "10",
    #     "num_02": "13",
    #     "num_03": "14",
    #     "num_04": "19",
    #     "num_05": "35",
    #     "num_06": "40"
    # })
    input_y = convertIndex([
        {
            "num_01": "10",
            "num_02": "13",
            "num_03": "14",
            "num_04": "19",
            "num_05": "35",
            "num_06": "40"
        },
        {
            "num_01": "04",
            "num_02": "06",
            "num_03": "13",
            "num_04": "25",
            "num_05": "31",
            "num_06": "41"
        },
        {
            "num_01": "07",
            "num_02": "10",
            "num_03": "14",
            "num_04": "21",
            "num_05": "26",
            "num_06": "37"
        }
    ])
    print("input_y:", input_y)
    processed_input_y = preprocess_input_y(input_y)
   
    print("processed_input_y:", processed_input_y)

if __name__ == "__main__":
    main()
