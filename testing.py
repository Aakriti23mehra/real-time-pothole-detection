import cv2 as cv
import numpy as np

# Correct path using raw string (prefix with r)
image = cv.imread(r'C:\Users\mehra\OneDrive\Documents\Desktop\Detection_of_Potholes\Images_CV\images.jpeg')
image_gray  = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

# to display an image 

cv.imshow("gray_scale_image ", image_gray)
cv.waitKey(0)
cv.destroyAllWindows()