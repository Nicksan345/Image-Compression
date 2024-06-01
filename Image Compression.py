import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Construct the file path using os.path.join
file_path = os.path.join("image.jpg")
#"C:\Users\nicks\OneDrive\Pictures\Tropics26325_rectangle.jpg""C:\Users\nicks\OneDrive\Pictures\pict.png"
# Load the image
image = cv2.imread(file_path)

# Check if the image is loaded successfully
if image is None:
    print("Error: Unable to load the image.")
    exit()

# Convert from BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get the shape of the image
h, w, c = image.shape

# Reshape the image to be a list of pixels
image_reshaped = image.reshape((-1, 3))

# Define the number of clusters (colors)
k = 5 # You can change this value

# Apply K-means
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(image_reshaped)

# Get the cluster centers (color palette)
centers = kmeans.cluster_centers_.astype('uint8')

# Get the labels for each pixel
labels = kmeans.labels_

# Recreate the compressed image
compressed_image = centers[labels].reshape((h, w, c))

# Plot the original and compressed images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(compressed_image)
plt.title('Compressed Image with K-means')
plt.axis('off')

plt.show()
