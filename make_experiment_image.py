"""
create blank images to understand what does it mean
to have standard deviation and mean of some image
"""
import numpy as np
import cv2

blank_image = np.zeros((300,300,3), np.uint8)
cv2.imwrite('img_0.png', blank_image)

white_image = np.ones((300,300,3), np.uint8)
white_image.fill(100)
cv2.imwrite('img_100.png', white_image)

white_image = np.ones((300,300,3), np.uint8)
white_image.fill(200)
cv2.imwrite('img_200.png', white_image)

