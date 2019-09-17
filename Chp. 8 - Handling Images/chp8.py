# Chapter 8 - Handling Images
# 8.0 Introduction
"""
Image classification is one of the most exciting areas of machine learning. The ability of computers to recognize 
patterns and objectsfrom imagesis an incredibly powerful tool in our toolkit. However, before we can apply machine 
learning to images, we often first need to transform the raw images to features usable by our learning algorithms.

To work with images, we will use the Open Source Computer Vision Library (OpenCV). While there are a number of good 
libraries out there, OpenCV is the most popular and documented library for handling images. On of the biggest hurdles
to using OpenCV is installing it. However, fortunately if we are using Python 3 (at the time of publication OpenCV 
does not work with Python 3.6+), we can use Anaconda's package manager tool conda to install OpenCV in a single line
of code in our terminal:

'>conda install --channel https://conda.anaconda.org/menpo opencv3'

Afterward, we can check the installation by opening a notebook, importing OpenCV, and checking the version number (3.1.0):

'import cv2'

'cv2.__version__'

If installing OpenCV using conda does not work, there are many guides onling.
Finally, throughout this chapter we will use a set of images as examples, which are available to download on GitHub
(https://github.com/chrisalbon/simulated_datasets).
"""

# 8.1 Loading Images
# When you want to load an image for preprocessing.
# Use OpenCV's 'imread':
# Load libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image as grayscale
image = cv2.imread("simulated_datasets-master/images/plane.jpg", cv2.IMREAD_GRAYSCALE)

# If we want to view the image, we can use the Python plotting library Matplotlib:
# Show image
plt.imshow(image, cmap= "gray"), plt.axis("off")
plt.show()

# Discussion: (http://bit.ly/2Fws76E & http://bit.ly/2FxZjKZ)
# Fundamentally, images are data and when we use 'imread' we convert that data into a data type we are very familiar 
# with - NumPy array:
# Show data type
type(image)

# We have transformed the image into a matrix whose elements correspond to individual pixels. We can even take a look
# at the vales of the matrix:
# Show image data
image

# The resolution of our image was 3600 X 2270, the exact dimensions of our matrix:
# Show dimensions
image.shape

# What does each element in the matrix actually represent? In grayscale images, the value of an individual element is
# the pixel intensity. Intensity values range from black (0) to white (255). For example, the intensity of the 
# top-rightmost pixel in our image has a value of 140:
# Show firt pixel
image[0, 0]

# In the matrix, each element contains three values corresponding to blue, green, red values (BGR):
# Load image in color
image_bgr = cv2.imread("simulated_datasets-master/images/plane.jpg", cv2.IMREAD_COLOR)

# Show pixel
image_bgr[0, 0]

# One small caveat: by default OpenCV uses BGR, but many image applications - including Matplotlib - use red, green,
# blue (RGB), meaning the read and the blue are swapped. To properly display OpenCV color images in Matplotlib, we 
# need to first convert the color to RGB (apologies for the hard copy readers):
# Convert to RGB
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Show image
plt.imshow(image_rgb), plt.axis("off")
plt.show()

# 8.2 Saving Images
# When you want to save an image for preprocessing.
# Use OpenCV's 'imwrite':
# Load libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image as grayscale
image = cv2.imread("simulated_datasets-master/images/plane.jpg", cv2.IMREAD_GRAYSCALE)

# Save image
cv2.imwrite("new_images/plane_new.jpg", image)

# Discussion:
# OpenCV's 'imwrite' saves images to the filepath specified. The format of the image is defined by the filename's 
# extension (.jpg, .png, etc.). One behavior to be careful about: 'imwrite' will overwrite existing files without 
# outputting an error or asking for confirmation.

# 8.3 Resizing Images
# When you to resize an image for further preprocessing.
# Use 'resize' to change the size of an image:
# Load libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image as grayscale
image = cv2.imread("simulated_datasets-master/images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Resize image to 50 pixels by 50 pixels
image_50x50 = cv2.resize(image, (50, 50))

# View image
plt.imshow(image_50x50, cmap = "gray"), plt.axis("off")
plt.show()

# Discussion:
# Resizing images is a common task in image preprocessing for two reasons. First images come in all shaoes and sizes,
# and to be usable as features, images must have the same dimensions. This standardization of image size does come
# with costs, however; images are matrices of information and when we reduce the size of the image we are reducing 
# that matrix and the information it contains. Second, machine learning can require thousands or hundreds of thousands
# of images. When those images are very large they can take up a lot of memory, and by resizing them we can 
# dramatically reduce memory usage. Some common image sizes for machine learning are 32 x 32, 64 x 64, 96 x 96, and
# 256 x 256.

# 8.4 Cropping Images
# When you want to remove the outer portion of the image to change its dimensions.
# The image is encoded as a two-dimensional NumPy array, so we can crop the image easily by slicing the array:
# Load libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image in grayscale
image = cv2.imread("simulated_datasets-master/images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Select first half of the columns and all rows
image_cropped = image[:,:128]

# Show image
plt.imshow(image_cropped, cmap = "gray"), plt.axis("off")
plt.show()

# Discussion (http://bit.ly/2FrVNBV)
# Since OpenCV represents images as a matrix of elements, by selecting the rows and columns we want to keep we are 
# able to easily crop the image. Cropping can be particularly useful if we know that we only want to keep a certain
# part of every image. For example, if our imagescome from a stationary security camera we can crop all the images
# so they only contain an area of interest.

# 8.5 Blurring Images
# When you want to smooth out an image.
# To blur an image, each pixel is transformed to be the average value of its neighbors. This neighbor and the 
# operation performed are mathematically represented as a kernel (don't worry if you don't know what a kernel is).
# The size of this kernel determines the amount of blurring, with larger kernels producing smoother images. Here we
# blur an image by averaging the values of a 5 x 5 kernel around each pixel:
# Load libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image as grayscale
image = cv2.imread("simulated_datasets-master/images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Blur image
image_blurry = cv2.blur(image, (5, 5))

# Show image
plt.imshow(image_blurry, cmap = "gray"), plt.axis("off")
plt.show()

# To highlight the effect of kernel size, here is the same blurring with a 100 x 100 kernel:
# Blur image
image_very_blurry = cv2.blur(image, (100, 100))

# Show image
plt.imshow(image_very_blurry, cmap = "gray"), plt.axis("off")
plt.show()

# Discussion
# Kernels are widely used in image processing to do everything from sharpening to edge detection, and will come up 
# repeatedly in this chapter. The blurring kernel we used looks like this:
# Create kernel
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

# See also:
# ~ Image Kernels Explained Visually: (http://setosa.io/ev/image-kernels)
# ~ Common Image Kernels: (http://bit.ly/2FxZCFD)

# 8.6 Sharpening Images
# When you want to sharpen an image.
# Create a kernel that highlights the target pixel. Then apply it to the image using 'filter2D':
# Load libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image as grayscale
image = cv2.imread("simulated_datasets-master/images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Create kernel
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

# Sharpen Image
image_sharp = cv2.filter2D(image, -1, kernel)

# Show image
plt.imshow(image_sharp, cmap = "gray"), plt.axis("off")
plt.show()

# Discussion
# Sharpening works similarly to blurring, except instead of using a kernel to average the neighboring values, we 
# constructed a kernel to highlight the pixel itself. The resulting effect makes contrasts in edges stand out more in 
# the image.

# 8.7 Enhancing Contrast
# When you want to increase the contrast between pixels in an image.
# Histogram equalization is a tool for image processing that can make objects and shapes stand out. When we have a 
# grayscale image, we can apply OpenCV's 'equallzeHist' directly on the image:
# Load libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image
image = cv2.imread("simulated_datasets-master/images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Enhance image
image_enhanced = cv2.equalizeHist(image)

# Show image
plt.imshow(image_enhanced, cmap = "gray"), plt.axis("off")
plt.show()

# However, when we have a color image, we first need to convert the image to the YUV color format. The Y is the luma,
# or brightness, and U and V denote the color. After the conversion, we can apply 'equalizeHist' to the image and 
# then convert it back to BGR or RGB:

# Load image
image_bgr = cv2.imread("simulated_datasets-master/images/plane.jpg")

# Convert to YUV
image_yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)

# Apply histogram equalization
image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])

# Convert to RGB
image_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)

# Show image
plt.imshow(image_rgb), plt.axis("off")
plt.show()

# Discussion
# While a detailed explanation of how histogram equalization works is beyond the scope of this book, the short 
# explanation is that it transforms the image so that it uses a wider range of pixel intensities.

# While the resulting image often does not look "realistic", we need to remember that the image is just a visual 
# representation of the underlying data. If histogram equalization is able to make objects of interest more 
# distinguishable from other objects or backgrounds (which is not always the case), then it can be a valuable 
# addition to our image preprocessing pipeline.

# 8.8 Isolating Colors
# When you want to isolate a color in an image.
# Define a range of colors and then apply a mask to the image:
# Load libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image
