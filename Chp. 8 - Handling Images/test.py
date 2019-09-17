import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image as grayscale
image = cv2. imread("simulated_datasets-master/images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Blur image
image_blurry = cv2.blur(image, (5, 5))

# Show image
plt.imshow(image_blurry, cmap = "gray"), plt.axis("off")
plt.show()


image_very_blurry = cv2.blur(image, (100, 100))

# Show image
plt.imshow(image_very_blurry, cmap = "gray"), plt.axis("off")
plt.show()

kernel = np.ones((5, 5)) / 25.0

# Show kernel
print(kernel)

# The center element in the kernel is the pixel being examined, while the remaining elements are its neighbors. Since
# all elements have the same value (normalized to add up to 1), each has an equal say in the resulting value of the 
# pixel of interest. We can manually apply a kernel to an image using 'filter2D' to produce a similar blurring 
# effect:
# Apply kernel
image_kernel = cv2.filter2D(image, -1, kernel)

# Show image
plt.imshow(image_kernel, cmap = "gray"), plt.axis("off")
plt.show()