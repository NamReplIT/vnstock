import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

y='1-0::2-8::3-4::4-12::5-20::6-18'

# Function to calculate similarity between two 'y' values
def calculate_similarity(y1, y2, weights=None):
    components1 = [int(x.split('-')[1]) for x in y1.split('::')]
    components2 = [int(x.split('-')[1]) for x in y2.split('::')]
    if weights is None:
        weights = [1] * len(components1)
    similarity = sum(w * (i == j) for i, j, w in zip(components1, components2, weights)) / sum(weights)
    return similarity

# Function to calculate near match similarity between two 'y' values
def calculate_near_match_similarity(y1, y2, threshold=2):
    components1 = [int(x.split('-')[1]) for x in y1.split('::')]
    components2 = [int(x.split('-')[1]) for x in y2.split('::')]
    similarity = sum(1 for i, j in zip(components1, components2) if abs(i - j) <= threshold) / len(components1)
    return similarity

# Function to perform cluster analysis on 'y' values
def perform_cluster_analysis(data, n_clusters=5):
    y_values = data['y'].apply(lambda y: [int(x.split('-')[1]) for x in y.split('::')])
    y_matrix = np.array(y_values.tolist())
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(y_matrix)
    data['cluster'] = kmeans.labels_
    return data

# Loading the pair.json data
def load_data(filepath):
    pair_data = pd.read_json(filepath)
    pair_data['x'] = pd.to_datetime(pair_data['x'])
    return pair_data

# Additional function to predict the next date's 'y' value
def predict_next_y_value(data, similar_date):
    # Find the index of the similar date
    similar_date_idx = data.index[data['x'] == similar_date].tolist()
    # Check if the next date exists in the dataset
    if similar_date_idx and similar_date_idx[0] + 1 < len(data):
        next_date = data.iloc[similar_date_idx[0] + 1]
        return next_date['x'], next_date['y']
    else:
        return None, None
    
# Main analysis function
def main():
    filepath = './pair.json'  # Replace with the actual file path
    pair_df = load_data(filepath)

    # Cluster analysis
    n_clusters = 5  # Number of clusters; can be adjusted
    clustered_data = perform_cluster_analysis(pair_df, n_clusters)

    # Provided 'y' value for analysis
    provided_y_value = y
    
    # Calculating similarity with a weighted approach
    weights = [5, 4, 3, 2, 1, 1]  # Weights can be adjusted
    pair_df['weighted_similarity'] = pair_df['y'].apply(lambda y: calculate_similarity(provided_y_value, y, weights))

    # Finding the most similar date
    most_similar_date = pair_df.loc[pair_df['weighted_similarity'].idxmax()]
    print('Most similar date (weighted):', most_similar_date['x'])

    # Near match similarity calculation
    near_match_threshold = 2  # Threshold for near matches; can be adjusted
    pair_df['near_match_similarity'] = pair_df['y'].apply(lambda y: calculate_near_match_similarity(provided_y_value, y, near_match_threshold))

    # Finding the most similar date with near match criteria
    most_similar_date_near_match = pair_df.loc[pair_df['near_match_similarity'].idxmax()]
    print('Most similar date (near match):', most_similar_date_near_match['x'])



    # Print the most similar date with similarity percentage (weighted)
    most_similar_date_idx = pair_df['weighted_similarity'].idxmax()
    print('Most similar date (weighted):', pair_df.at[most_similar_date_idx, 'x'])
    print('Weighted similarity percentage:', pair_df.at[most_similar_date_idx, 'weighted_similarity'] * 100, '%')

    # Print the most similar date with similarity percentage (near match)
    most_similar_date_near_match_idx = pair_df['near_match_similarity'].idxmax()
    print('Most similar date (near match):', pair_df.at[most_similar_date_near_match_idx, 'x'])
    print('Near match similarity percentage:', pair_df.at[most_similar_date_near_match_idx, 'near_match_similarity'] * 100, '%')


    # Predict the 'y' value for the next date after the most similar one
    next_date, next_y_value = predict_next_y_value(pair_df, pair_df.at[most_similar_date_idx, 'x'])
    if next_date:
        print('Next date:', next_date)
        print('Suggested \'y\' value for next date:', next_y_value)
    else:
        print('No subsequent date available in the dataset.')

# Entry point of the script
if __name__ == "__main__":
    main()
