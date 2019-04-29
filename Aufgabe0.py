import numpy as np
import cv2
import requests



sift = cv2.xfeatures2d.SIFT_create()
cap = cv2 . VideoCapture(0)
req = requests.get('https://cdn-images-1.medium.com/freeze/max/1000/1*nrhQ7gilqLbYai0N4bPUQ.jpeg?q=20')
arr = np.asarray(bytearray(req.content) , dtype= np . u i n t 8 )
img = cv2.imdecode(arr,−1) # 'Load it as it is '
while(True):
    # Capture  frame -by-frame
    ret, frame = cap.read ()

    # Our  operations  on the  frame  come  here
	try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    if cv2.waitKey (1) & 0xFF == ord ('q') :
        break

    # Display  the  resulting  frame
    cv2.imshow ('frame', gray)
	
#When  everything  done , release  the  capture
cap.release()
cv2.destroyAllWindows()