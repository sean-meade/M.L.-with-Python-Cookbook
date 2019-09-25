# 8.15 Encoding Color Histograms as Features
# When you want to create a set of features representing the colors appearing in an image.
# Compute the histograms for each color channel:
# Load libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image_bgr = cv2.imread("simulated_datasets-master/images/plane_256x256.jpg", cv2.IMREAD_COLOR)

# Convert to RGB
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Create a list for feature values
features = []

# Calculate the histogram for each color channel
colors = ("r", "g", "b")

# For each channel: calculate histogram and add to feature value list
for i, channel in enumerate(colors):
    histogram = cv2.calcHist([image_rgb], # Image
                             [i], # Index of channel
                             None, # No mask
                             [256], # Histogram size
                             [0, 256]) # Range
    features.extend(histogram)

# Create a vector for an observation's feature values
observation = np.array(features).flatten()

# Shoow the observation's value for the first five features
colors = ("r", "g", "b")

# For each channel: calculate histogram, make plot
for i, channel in enumerate(colors):
    histogram = cv2.calcHist([image_rgb], # Image
                             [i], # Index of channel
                             None, # No mask
                             [256], # Histogram size
                             [0, 256]) # Range
    plt.plot(histogram, color = channel)
    plt.xlim([0, 256])

# Show plot
plt.show()