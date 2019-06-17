# Example a)i)
def filter_matches(keypoints1, keypoints2, matches):
    keys1 = np.float32([keypoints1[i.queryIdx].pt for i in matches])
    keys2 = np.float32([keypoints2[i.trainIdx].pt for i in matches])
    return keys1, keys2

# Example a)ii)
(H, status) = cv2.findHomography(key1, key2, cv2.RANSAC, 4.0)

# Example b)
def warp_image(img, homograpy):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    newImage = cv2.warpPerspective(img, homograpy, (img.shape[1]*2, img.shape[0]))
    blackrows = np.any(newImage[...,-1], axis=0)
    first_black_row = np.where(blackrows[::-1])[0][0]
    return newImage[:,:-first_black_row,:3]
