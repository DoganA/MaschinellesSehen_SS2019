import cv2
import numpy as np


# Load image and convert to gray and floating point
img = cv2.imread('./images/Lenna.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
# Define parameter
k = 0.04
threshold_factor = 0.01

# a) Define sobel filter and use cv2.filter2D (or use cv2.Sobel) to filter the grayscale image

# YOUR CODE HERE

# b) Compute I_xx, I_yy, I_xy and sum over all 3x3 neighbors to compute
# entries of the matrix M = \sum_{3x3} [ I_xx Ixy; Ixy Iyy ]
# Note1: this results again in 3 images sumIxx, sumIyy, sumIxy
# Hint: to sum the neighbor values you can again use cv2.filter2D to do this efficiently

# YOUR CODE HERE

# c)+d) Compute the determinat and trace of M using sumGxx, sumGyy, sumGxy. With det(M) and trace(M)
# you can compute the resulting image containing the harris corner responses with
# R = ...

# YOUR CODE HERE

# Filter the harris 'image' R with 'R > threshold*R.max()'
# 'R > threshold*R.max()' will give you a boolean mask where values are above the threshold.
# That boolean mask are the corner pixel you want to use.
# In order to make the rest of the code work out of the box name that boolean mask harris_thres:
# harris_thres = ...

# YOUR CODE HERE


# The OpenCV implementation looks like this - please do not change
harris_cv = cv2.cornerHarris(gray,3,3,k)
# intialize in black - set pixels with corners in with
harris_cv_thres = harris_cv > threshold_factor*harris_cv.max()


# please leave this - adjust variable name if desired
print("====================================")
print("DIFF:", np.sum(np.absolute(harris_thres.astype(int) - harris_cv_thres.astype(int))))
print("====================================")


## To test your code uncomment the following lines:

# just for debugging to create such an image as seen
# in the assignment figure.
# img[R>threshold_factor*R.max()]=[255,0,0]

# cv2.imwrite("Harris_own.png", harris_thres)
# cv2.imwrite("Harris_cv.png", harris_cv_thres)
# cv2.imwrite("Image_with_Harris.png", img)

# while True:
#     cv2.imshow('harris',harris_thres)
#     cv2.imshow('harris_cv',harris_cv_thres)
#     cv2.imshow('harris',harris_thres*1.)
#     cv2.imshow('harris_cv',harris_cv_thres*1.)
#     ch = cv2.waitKey(0)
#     if ch == 27:
#         break
