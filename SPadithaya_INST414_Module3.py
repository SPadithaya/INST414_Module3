import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
health_data = pd.read_csv("Disease_symptom_and_patient_profile_dataset.csv")

# Inspect dataset
print("Dataset Sample:")
print(health_data.head())

# Combine symptom columns into a single text column
symptom_columns = ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]
health_data["symptoms_combined"] = health_data[symptom_columns].apply(lambda x: ' '.join(x.index[x == "Yes"]), axis=1)

# Convert symptoms into TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words="english")
symptom_vectors = vectorizer.fit_transform(health_data["symptoms_combined"])

# Compute similarity matrix
similarity_matrix = cosine_similarity(symptom_vectors)

# Function to get top 10 most similar conditions
def get_similar_conditions(condition_name, top_n=10):
    if condition_name not in health_data["Disease"].values:
        return f"Condition '{condition_name}' not found in dataset."
    
    idx = health_data.index[health_data["Disease"] == condition_name][0]
    similar_indices = similarity_matrix[idx].argsort()[::-1][1:]  # Get sorted similarity scores
    similar_conditions = []
    
    for i in similar_indices:
        condition = health_data.iloc[i]["Disease"]
        if condition not in similar_conditions:  # Avoid duplicates
            similar_conditions.append(condition)
        if len(similar_conditions) == top_n:
            break
            
    return similar_conditions

# Example queries: Comparing conditions to Influenza
query_conditions = ["Influenza", "Pneumonia", "Common Cold"]

for query in query_conditions:
    print(f"Top 10 conditions similar to {query}:")
    print(get_similar_conditions(query))
    print("\n")
