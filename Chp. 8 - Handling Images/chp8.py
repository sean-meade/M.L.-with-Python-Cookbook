# Chapter 8 - Handling Images
# Adapted from the book: Machine Learning with Python Cookbook by Chris Albon
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
image_bgr = cv2.imread('simulated_datasets-master/images/plane_256x256.jpg')

# Convert BGR to HSV
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

# Define range of blue values in HSV
lower_blue = np.array([50, 100, 50])
upper_blue = np.array([130, 255, 255])

# Create mask
mask = cv2.inRange(image_hsv, lower_blue, upper_blue)

# Mask image
image_bgr_masked = cv2.bitwise_and(image_bgr, image_bgr, mask = mask)

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image_bgr_masked, cv2.COLOR_BGR2RGB)

# Show image
plt.imshow(image_rgb), plt.axis("off")
plt.show()

# Discussion
# Isolating colors in OpenCV is straightforward. First we convert an image into HSV (hue, saturation, and value).
# Second, we define a range of vlaues we want to isolate, which is probably the most difficult and time- consuming 
# part. Third, we create a mask for the image (we will only keep the white space):
# Show image
plt.imshow(mask, cmap = "gray"), plt.axis("off")
plt.show()

# Finally, we apply the mask to the image using 'bitwise_and' and convert to our desired output format.

# 8.9 Binarizing Images
# When you're given an image and you want to output a simplified version.
# Thresholding is the process of setting pixels with intensity greater than some value to be white and less than the 
# value to be black. A more advanced technique is adaptive thresholding, where the threhold value for a pixel is 
# determined by the pixel intensities of its neighbors. This can be helpful when lighting conditions change over 
# different regions of an image:
# Load libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image as grayscale
image_grey = cv2.imread("simulated_datasets-master/images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Apply adaptive thresholding
max_output_value = 255
neighborhood_size = 99
subtract_from_mean = 10
image_binarized = cv2.adaptiveThreshold(image_grey,
                                        max_output_value,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY,
                                        neighborhood_size,
                                        subtract_from_mean)

# Show image
plt.imshow(image_binarized, cmap = "gray"), plt.axis("off")
plt.show()

# Discussion:
# Our solution has four important arguments in 'adaptiveThreshold'. 'max_output_value simply determines the maximum 
# intensity of the output pixel intensities. 'cv2.ADAPTIVE_THRESH_GAUSSIAN_C' sets a pixel's threshold to be a weighted 
# sum of the neighboring pixel intensities. The weights are determined by a Gaussian window. Alternatively we could 
# set the threshold to simply the mean of the neighboring pixels with 'cv2.ADAPTIVE_THRESH_MEAN_C':

# Apply cv.ADAPTIVE_THRESH_MEAN_C
image_mean_threshold = cv2.adaptiveThreshold(image_grey,
                                             max_output_value,
                                             cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY,
                                             neighborhood_size,
                                             subtract_from_mean)

# Show image
plt.imshow(image_mean_threshold, cmap = "gray"), plt.axis("off")
plt.show()

# The last two parameters are the block size (the size of the neighborhood used to determine a pixel's threshold) and
# a constant subtracted from the calculated threshold (used to manually fine-tune the threshold).

# A major benefit of thresholding is denoising an image - keeping only the most important elements. For example, 
# thresholding is often applied to photos of printed text to isolate the letters from the page.

# 8.10 Removing Backgrounds
# For when you want to isolate the foreground of an image.
# Mark a rectangle around the desired foreground, then run the GrabCut algorithm:
# Load library
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image and convert to RGB
image_bgr = cv2.imread('simulated_datasets-master/images/plane_256x256.jpg')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Rectangle values: start x, start y, width, height
rectangle = (0, 56, 256, 150)

# Craete initial mask
mask = np.zeros(image_rgb.shape[:2], np.uint8)

# Create temporary arrays used by grabCut
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# Run grabCut
cv2.grabCut(image_rgb, # Our image
            mask, # The mask
            rectangle, # Our rectangle
            bgdModel, # Temporary array for background
            fgdModel, # Temporary array for background
            5, # Number of iterations
            cv2.GC_INIT_WITH_RECT) # Initiative using our rectangle

# Create mask where sure and likely backgrounds set to 0, otherwise 1
mask_2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Multiply image with new mask to subtract background
image_rgb_nobg = image_rgb * mask_2[:, :, np.newaxis]

# Show image
plt.imshow(image_rgb_nobg), plt.axis("off")
plt.show()

# Discussion
# The first thing we notice is that even though GrabCut did a pretty good job, there are still areas of background 
# left in the image. We could go back and manually mark those areas as background, but in the real world we have 
# thousands of images and manually fixing them individually is not feasible. Therefore, we would do well by simply 
# accepting that the image data will still contain some background noise.

# In our solution, we start out by marking a rectangle around the area that contains the foreground. GrabCut assumes
# everything outside this rectangle to be backgroundand and uses that information to figure out what is likely to be
# background inside the square (to learn how the algorithm does this, check out the external resources at the end of 
# this solution). Then a mask is created that denotes the difference definitely/likely baclground/foreground regions:
# Show mask
plt.imshow(mask, cmap = "gray"), plt.axis("off")
plt.show()

# The black region is the area outside our rectangle that is assumed to be definitely background. The gray area is 
# what GrabCut considered likely background, while the white area is likely foreground.

# This mask is then used to create a second mask that merges the black and gray regions:
# Show mask
plt.imshow(mask_2, cmap = "gray"), plt.axis("off")
plt.show()

# The second mask is then applied to the image so that only the foreground remains.

# 8.11 Detecting Edges
# When you want to find the edges of an image.
# Use an edge detection technique like the Canny edge detector:
# Load libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image as grayscale
image_gray = cv2.imread("simulated_datasets-master/images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Calculate median intensity
median_intensity = np.median(image_gray)

# Set thresholds to be one standard deviation above and below median intensity
lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
upper_threshold = int(min(255, (1.0 - 0.33) * median_intensity))

# Apply canny edge detector
image_canny = cv2.Canny(image_gray, lower_threshold, upper_threshold)

# Show image
plt.imshow(image_canny, cmap = "gray"), plt.axis("off")
plt.show()

# Discussion
# Edge detection is a major topic of interest in computer vision. Edges are important because they are areas of high
# information. For example, in our image on patch of sky looks very much like another and is unlikely to contain 
# unique or interesting information. However, patches where the background sky meets the airplane contain a lot of
# information (e.g., an object's shape). Edge detection allows us to remove low-information areas and isolate the 
# areas of images containing the most information.

# There are many edge detection techniques (Sobel filters, Laplacian edge detector, etc.). However, our solution uses
# the commonly used Canny edge detector. How the Canny detector works is too detailed for this book, but there is one
# point that we need to address. The Canny detector requires two parameters denoting low and high gradient threshold
# values. Potential edge pixels between the low and high thresholds are considered weak edge pixels, while those above
# the high threshold are considered strong edge pixels. OpenCV'S Canny method includes the low and high thresholds as
# required parameters. In our solutions, we set the lower and upperthresholds to be one standard deviation below and
# above the images median pixel intensity. However, there are often cases when we might get better results if we used
# a good pair of low and high threshold values through manual trial and error using a few images before running Canny
# on our entire collection of images.

# See Also:
#   - Canny Edge Detector (http://bit.ly/2FzDXNt)
#   - Canny Edge Detection Auto Thresholding (http://bit.ly.2nmQERq)

# 8.12 Detecting Corners
# When you want to detect the corners in an image.
# Use OpenCV's implementation of the Harris corner detector, 'cornerHarris':
# Load libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image as grayscale
image_bgr = cv2.imread("simulated_datasets-master/images/plane_256x256.jpg")
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
image_gray = np.float32(image_gray)

# Set corner detector parameters
block_size = 2
aperture = 29
free_parameter = 0.04

# Detect corners
detector_responses = cv2.cornerHarris(image_gray,
                                      block_size,
                                      aperture,
                                      free_parameter)

# Large corner markers
detector_responses = cv2.dilate(detector_responses, None)

# Only keep detector responses greater than threshold, mark as white
threshold = 0.02
image_bgr[detector_responses >
          threshold *
          detector_responses.max()] = [255, 255, 255]

# Convert to grayscale
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# Show Image
plt.imshow(image_gray, cmap = "gray"), plt.axis("off")
plt.show()

# Discussion
# The Harris corner detector is a commonly used method of detecting the intersection of two edges. Our interest in 
# detecting corners is motivated by the same reason as for deleting edges: corners are points of high information. A
# complete explanation of the Harris corner detector is available in the external resources at the end of this recipe,
# but a simplified explanation is that it looks for windows (also called neighborhoods or patches) where small 
# movements of the window (imagine shaking the window) creates big changes in the contents of the pixels inside the
# window. 'cornerHarris' contains three important parameters that we can use to control the edges detected. First, 
# 'block_size' is the size of the neighbor around each pixel used for corner detection. Second, aperture is the size
# of the Sobel kernel used (don't worry if you don't know what that is), and finally there is a free parameter where
# larger values correspond to identifying softer corners.

# The output is a grayscale image depicting potential corners:
# Show potential corners
plt.imshow(detector_responses, cmap = "gray"), plt.axis("off")
plt.show()

# We then apply thresholding to keep only the most likely corners. Alternatively, we can use a similar detector, the 
# Shi-Tomasi corner detector, which works in a similar way to the Harris detector ('goodFeaturesToTrack') to identify
# a fixed number of strong corners, 'goodFeaturesToTrack' takes three major parameters - the number of corners to 
# detect, the minimum quality of the corner (0 to 1), and the minimum Euclidean distance between corners:

# Load images
image_bgr = cv2.imread("simulated_datasets-master/images/plane_256x256.jpg")
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# Number of corners to detect
corners_to_detect = 10
minimum_quality_score = 0.05
minimum_distance = 25

# Detect corners
corners = cv2.goodFeaturesToTrack(image_gray,
                                  corners_to_detect,
                                  minimum_quality_score,
                                  minimum_distance)
corners = np.float32(corners)

# Draw white circle at each corner
for corner in corners:
    x, y = corner[0]
    cv2.circle(image_bgr, (x,y), 10, (255, 255, 255), -1)

# Convert to grayscale
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# Show image
plt.imshow(image_rgb, cmap = "gray"), plt.axis("off")
plt.show()

# See Also
#   - OpenCV's cornerHarris (http://bit.ly/2HQXwz6)
#   - OpenCV's goodFeaturesToTrack (http://bit.ly/2HRSVwF)

# 8.13 Creating Features for Machine Learning
# When you want to convert an image into an observation for machine learning.
# Use NumPy's 'flatten' to convert the multidimensional array containing an image's data into a vector containing the
# observation's values:
# Load libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image as grayscale
image = cv2.imread("simulated_datasets-master/images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Resize image to 10 pixels by 10 pixels
image_10x10 = cv2.resize(image, (10, 10))

# Convert image data to one-dimensional vector
image_10x10.flatten()

# Discussion
# Images are presented as a grid of pixels. If an image is in grayscale, each pixel is presented by one value (i.e.,
# pixel intensity: 1 if white, 0 if black). For example, imagine we have a 10 x 10 - pixel image:
plt.imshow(image_10x10, cmap = 'gray'), plt.axis("off")
plt.show()

# In this case the dimensions of the images data will be 10 x 10:
image_10x10.shape

# And if we flatten the array, we get a vector of length 100 (10 multiplied by 10):
image_10x10.flatten().shape

# This is the feature data for our image that can be joined with the vectors from other images to create the data we 
# will feed to our machine learning algorithms.

# If the image is in color, instead of each pixel being represented by one value, it is represented by multiple values
# (most often three) representing the channels (red, green, blue, etc.) that blend to make the final color of that 
# pixel. For this reason, if our 10 x 10 image is in color, we will have 300 feature values for each observation:
# Load image in color
image_color = cv2.imread("simulated_datasets-master/images/plane_256x256.jpg", cv2.IMREAD_COLOR)

# Resize image to 10 pixels by 10 pixels
image_color_10x10 = cv2.resize(image_color, (10, 10))

# Convert image data to one-dimensional vector, show dimensions
image_color_10x10.flatten().shape

# One if the major challenges of image processing and computer vision is that since every pixel location in a 
# collection of images is a feature, as the images get larger, the number of features explodes:
# Load image in grayscale
image_256x256_gray = cv2.imread("simulated_datasets-master/images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

# Convert image data to one-dimensional vector, show dimensions
image_256x256_gray.flatten().shape

# And the number of featrures only intensifies when the image is in color:
# Load image in color
image_256x256_color = cv2.imread("simulted_datasets-master/images/plane_256x256.jpg", cv2.IMREAD_COLOR)

# Convert image data to one-dimensional vector, show dimensions
image_256x256_color.flatten().shape

# As the output shows, even a small color image has almost 200,000 features, which can cause problems when we are
# training our models because the number of features might far exceed the number of observations.

# This problem will motivate dimensionality strategies discussed in a later chapter, which attempt to reduce the 
# number of features while not losing excessive amounts of information contained in the data.

# 8.14 Encoding Mean Color as a Feature
# When you want a feature based on the colors of an image.
# Each pixel in an image is represented by the combination of multiple color channels (often three: red, green, and 
# blue). Calculate the mean red, green, and blue channel values for an image to make three color features representing
# the average colors in that image:
# Load libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image as RGB
image_bgr = cv2.imread("simulated_datasets-master/images/plane_256x256.jpg", cv2.IMREAD_COLOR)

# Calculate the mean of each channel
channels = cv2.mean(image_bgr)

# Swap blue and red values (making it RGB, not BGR)
observation = np.array([(channels[2], channels[1], channels[0])])

# Show mean channel values
observation

# We can view the mean channel values directly (apologies to printed book readers):
# Show image
plt.imshow(observation), plt.axis("off")
plt.show()

# Discussion
# The output is three feature values for an observation, one for each color channel in the image. These features can 
# be used like any other features in learning algorithms to classify images according to their colors.

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
observation[0:5]

# Discussion
# In the RGB color model, each color is the combination of three color channels (i.e., red, green, blue). In turn, 
# each channel can take on one of 256 values (represented by an integer between 0 and 255). For example, the 
# top-leftmost pixel in our image has the following channel values:
# Show RGB channel values
image_rgb[0, 0]

# A histogram is a representation of the distribution of values in data. Here is a simple example:
# Import pandas
import pandas as pd

# Create some data
data = pd.Series([1, 1, 2, 2, 3, 3, 3, 4, 5])

# Show the histogram
data.hist(grid = False)
plt.show()

# In this example, we have some data with two 1s, two 2s, three 3s, one 4, and one 5. In the histogram, each bar 
# represents the number of times each value (1, 2, etc.) appears in our data.

# We can apply this same technique to each of the color channels, but instead of five possible values we have 256 
# (the range of possible values for a channel). The x-axis represents the 256 possible channel values, and the y-axis
# represents the number of times a particular channel value appears across all pixels in an image:
# Calculate the histogram for each color channel
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

# As we can see in the histogram, barely any pixels contain the blue channel values between 0 and ~180, while many 
# pixels contain blue channel values between ~190 and ~210. This distribution of channel values is shown for all 
# three channels. The histogram, however, is not simply a visualization; it 256 features for each color channel, 
# making for 768 features representing the distribution of colors in an image.

# See Also:
#   - Histogram (https://en.wikipedia.org/wiki/Histogram)
#   - pandas Histogram documentation (http://bit.ly/2HT4Fz0)
#   - OpenCV Histogram tutorial (http://bit.ly/2HSyoYH)

