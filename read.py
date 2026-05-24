import cv2 as cv


image = cv.imread(r'C:\Users\mehra\OneDrive\Documents\Desktop\Detection_of_Potholes\Images_CV\images.jpeg')
alpha  = 2
beta = 20


enhanced_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)

cv.imshow("Original image", image)
cv.imshow("Enhanced image ", enhanced_image)

cv.waitKey(0)
cv.destroyAllWindows()
