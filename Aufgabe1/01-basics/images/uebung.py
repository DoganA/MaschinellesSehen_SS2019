import cv2
import numpy


lena = cv2.imread("lenna.png"); #öffnet fenster


cv2.imshow('lenna', lena)  #öffnet fenster, indem das bild angezeigt wird

print(lena.shape)
#lena_blau = lena[:,:,0]
#lena_gruen = lena[:,:,1]
#lena_rot = lena[:,:,2]

lena = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)

#cv2.adaptiveTreshold()
print(lena.shape)

cv2.imshow('Lena', lena)
#cv2.imshow('Lena Blau', lena_blau)
#cv2.imshow('Lena Grün', lena_gruen)
#cv2.imshow('Lena Rot', lena_rot)

while cv2.waitKey(100) != ord('q'): #kontrolliert alle 100 microsekunden, ob die taste q gedrückt wurde 
    pass