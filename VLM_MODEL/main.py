from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os


#Load embeddings
embedding_dir = "../output"

#Loops through .npy files in the output directory, loads them into a list, and keeps track of their names
embedding_files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]  
embeddings = []
image_names = []

#Load each embedding and store it in a list, along with the corresponding image name.
#Where embeddings are stored as numpy arrays used for analysis, and image names are stored as strings in a separate list.
for file in embedding_files:
    path = os.path.join(embedding_dir, file)
    embeddings.append(np.load(path))
    image_names.append(file)

try:
    embeddings = np.stack(embeddings)
    print("Loaded embeddings shape:", embeddings.shape)
except ValueError:
    print(f"Loaded {len(embeddings)} embeddings with varying shapes (kept as list)")



# Ensure embeddings are uniform for cosine similarity
# Find the most common shape
shapes = [emb.shape for emb in embeddings]
most_common_shape = max(set(shapes), key=shapes.count)
print(f"Most common embedding shape: {most_common_shape}")

# Keep only embeddings with the most common shape and flatten them
embeddings_flat = np.array([emb.flatten() for emb in embeddings if emb.shape == most_common_shape])
valid_names = [name for emb, name in zip(embeddings, image_names) if emb.shape == most_common_shape]

# Compute cosine similarity, calculates similarity of all embeddings 
similarity_matrix = cosine_similarity(embeddings_flat)

print(f"Cosine similarity matrix shape: {similarity_matrix.shape}")
print(f"Similarity of first 2 images: {similarity_matrix[0, 1]}")


#Clustering
from sklearn.cluster import KMeans
 
numm_clusters = 5
kmeans = KMeans(n_clusters=numm_clusters, random_state=42)
labels = kmeans.fit_predict(embeddings_flat)

for img, label in zip(valid_names, labels):
    print(f"{img} → Cluster {label}")


#anomaly detection using Isolation Forest, identifies outliers in the embedding space which may correspond to anomalous images.

from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.1, random_state=42)
anomaly_labels = iso.fit_predict(embeddings_flat)

for img, label in zip(valid_names, anomaly_labels):
    if label == -1:
        print(f"{img} → Anomaly detected")
    else:
        print(f"{img} → Normal")