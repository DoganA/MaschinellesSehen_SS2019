import cv2
import numpy as np


def render_virtual_object(img, quadrat_2d, quadrat_2d_transformed):

    # define vertices, edges and colors: In this Case a 3D Cube
    vertices = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                           [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5), (2, 6), (3, 7), (4, 5), (5, 6), (6, 7), (7, 4)]
    color_lines = (0, 0, 0)

    # define quadrat_ plane in 3D coordinates with z = 0
    quadrat_3d = np.float32([[quadrat_2d[0][0], quadrat_2d[0][1], 0], [quadrat_2d[1][0], quadrat_2d[1][1], 0], [quadrat_2d[2][0], quadrat_2d[2][1], 0], [quadrat_2d[3][0], quadrat_2d[3][1], 0]])

    h, w = img.shape[:2]
    # define intrinsic camera parameter
    # ADD HERE: K should be the value of YOUR intrinsic camera parameters from Assignment 1 (camera calibration)
    raise Exception("You have to set your intrinsic camera")
    # Example:
    # K = np.float64([[w, 0, 0.5*(w-1)],
    #                 [0, w, 0.5*(h-1)],
    #                 [0, 0, 1.0]])

    # ADD HERE: dist_coef should be YOUR distortion coefficients from Assignment 1 (camera calibration)
    raise Exception("You have to set your distortion coefficients")
    # Example:
    # dist_coef = np.array([0.2455461, -1.52216645, 0.0263676, -0.02353695, 2.98092726])

    # find object pose from 3D-2D point correspondences of the 3d quadrat using Levenberg-Marquardt optimization
    ret, rvec, tvec = cv2.solvePnP(quadrat_3d, quadrat_2d_transformed, K, dist_coef, False, cv2.SOLVEPNP_EPNP)

    # transform vertices: scale and translate form 0 - 1, in window size of the marker
    # quadrat_2d = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    cubeheight = quadrat_2d[0][0]-quadrat_2d[2][0]
    z_offset = 0
    scale = [quadrat_2d[2][0]-quadrat_2d[0][0], quadrat_2d[2][1]-quadrat_2d[0][1], cubeheight]
    trans = [quadrat_2d[0][0], quadrat_2d[0][1], z_offset]

    verts = scale * vertices + trans
    # returns a tuple that includes the transformed vertices as a first argument
    verts = cv2.projectPoints(verts, rvec, tvec, K, dist_coef)[0]
    verts = verts.reshape(-1, 2)
    # draw the Object
    for i, j in edges:
        (x_start, y_start), (x_end, y_end) = verts[i], verts[j]
        if min(int(x_start), int(y_start), int(x_end), int(y_end)) < 0:
            continue
        if max(int(x_start), int(x_end)) >= w:
            continue
        if max(int(x_start), int(x_end)) >= h:
            continue
        cv2.line(img, (int(x_start), int(y_start)), (int(x_end), int(y_end)), color_lines, 2)


def filter_matches(keypoints1, keypoints2, matches):
    keys1 = np.float32([keypoints1[i.queryIdx].pt for i in matches])
    keys2 = np.float32([keypoints2[i.trainIdx].pt for i in matches])
    return keys1, keys2


def render_quadrat_2d(img, quadrat_2d):
    # render quadrat_ in image plane and feature points as circle using cv2.polylines + cv2.circle
    cv2.polylines(img, [np.int32(quadrat_2d)], True, (0, 0, 255), 2)


def render_keypoints(img, points):
    for (x, y) in np.int32(points):
        cv2.circle(img, (x, y), 15, (0, 0, 255))


#-------------------------------------------------------------------------------
# INITALIZATION
#-------------------------------------------------------------------------------
# global constants
min_inlier = 10
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary


# initialize flann and SIFT extractor
flann = cv2.FlannBasedMatcher(index_params, search_params)
sift = cv2.xfeatures2d.SIFT_create()

# extract marker descriptors
marker = cv2.imread('./images/magazin-marker.jpg')
keypointsMarker, descriptorsMarker = sift.detectAndCompute(marker, None)

# get the size of the marker and form a quadrat_ in pixel coords np float array using w/h as the corner points
h, w = marker.shape[:2]
x0 = 0
y0 = 0
x1 = w
y1 = h
quadrat_2d = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
print("# kps: {}, descriptors: {}".format(len(keypointsMarker), descriptorsMarker.shape))


scaling_factor = 1
window_name = 'Computer Vision: AR Tracking'
cap = cv2.VideoCapture(0)
cv2.namedWindow(window_name)
while True:
    # keyhandling: ESC to shutdown
    ch = cv2.waitKey(1)
    if ch == 27:
        break

    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    # detect SIFT Features and compute Feature descriptors in camera image
    # and match with marker descriptor
    keypointsFrame, descriptorsFrame = sift.detectAndCompute(frame, None)
    matches = flann.knnMatch(descriptorsFrame,descriptorsMarker, k=2)

    # filter matches by distance [Lowe2004]
    matches = [match[0] for match in matches if len(match) == 2 and match[0].distance < match[1].distance * 0.75]

    # extract 2d points from matches data structure
    p0, p1 = filter_matches(keypointsFrame, keypointsMarker, matches)

    # we need at least 4 match points to find a homography matrix
    # otherwise continue
    if len(p0) < 4:
        cv2.imshow(window_name, frame[:,::-1])
        continue

    # find homography using p0 and p1, returning H and status
    # H: homography matrix
    # mask: mask of Outlier/Inlier
    H, mask = cv2.findHomography(p1, p0, cv2.RANSAC, 4.0)
    # take only inliers
    mask = mask.ravel() != 0
    p0, p1 = np.squeeze(p0[mask]), np.squeeze(p1[mask])
    # if we got less than min_inlier
    if len(p0) < min_inlier:
        cv2.imshow(window_name, frame[:,::-1])
        continue

    # render all inliers
    render_keypoints(frame, p0)
    # perspectiveTransform needs a 3-dimensional array
    quadrat_transformed = cv2.perspectiveTransform(np.array([quadrat_2d]), H)
    # transform back to 2D array
    quadrat_transformed = quadrat_transformed[0]

    # render the quadrat
    render_quadrat_2d(frame, quadrat_transformed)
    # render virtual object on top of quadrat
    render_virtual_object(frame, quadrat_2d, quadrat_transformed)

    # show the image
    cv2.imshow(window_name, frame[:,::-1])

# clean-up
cap.release()
cv2.destroyAllWindows()
