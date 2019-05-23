import cv2
import glob
from Queue import PriorityQueue

############################################################
#
#              Simple Image Retrieval
#
############################################################


# implement distance function
def distance(a, b):
    # YOUR CODE HERE
    pass


def create_keypoints(w, h, grid_size, keypointSize):
    keypoints = []

    # YOUR CODE HERE

    return keypoints


# 1. preprocessing and load
image_names = glob.glob('./images/db/*/*.jpg')

# 2. create keypoints on a regular 15x15 grid (cv2.KeyPoint(x, y, keypointSize), as keypoint size use 11):
keypoints = create_keypoints(w=256, h=256, grid_size=(15,15), keypointSize=11)

# 3. load each image and use the keypoints for each image to compute SIFT descriptors
#    for each keypoint. This computes one descriptor for each image. Save this descriptor.

# YOUR CODE HERE

# 4. use the three query input images to query the 'image database'.
#    Therefore extract the descriptor for a query Image and
#    compare the descriptor to each image in the database using the L2-norm
#    and save the result into a priority queue (q = PriorityQueue())

# YOUR CODE HERE

# 5. output (save and/or display) the query results in the order of smallest distance

# YOUR CODE HERE
