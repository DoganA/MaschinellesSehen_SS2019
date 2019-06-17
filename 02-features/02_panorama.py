import numpy as np
import cv2


def draw_matches(img1, img2, kp1, kp2, matches):
    """For each pair of points we draw a line between both images and circles,
    then connect a line between them. This is for visualization of which points are used
    to compute the homography only"""

    (h1, w1) = img1.shape[:2]
    (h2, w2) = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")
    vis[0:h1, 0:w1] = img1
    vis[0:h2, w1:] = img2

    for i in matches:
        idx2, idx1 = i.trainIdx, i.queryIdx
        # x - columns
        # y - rows
        (x1, y1) = kp1[idx1].pt
        (x2, y2) = kp2[idx2].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(vis, (int(x1), int(y1)), 4, (255, 255, 0), 1)
        cv2.circle(vis, (int(x2) + w1, int(y2)), 4, (255, 255, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(vis, (int(x1), int(y1)), (int(x2) + w1, int(y2)), (255, 0, 0), 1)
    return vis

############################################################
#
#                   Image Stitching
#
############################################################

# 1. load panorama images

# order of input images is important (from right to left)
imageList = [] # list of images


# 2. detect and compute SIFT features
def find_keypoints(image):
    # TODO: Return keypoints and descriptors
    ...

# 3. match keypoints using FLANN (or BFMatcher)
def match_keypoints(descriptors1, descriptors2):
    # 3.0 FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    # 3.1 Find FLANN Matches
    # TODO

    # 3.2 ratio test as per Lowe's paper (ration < 0.7) to filter out matches
    # TODO


# 4. for ever two adjacent images:
#   find keypoints
#   find matches
#   draw matches
#   save image
# TODO
