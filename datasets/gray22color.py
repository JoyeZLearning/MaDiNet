import cv2
import numpy as np


img = cv2.imread(r' ')


dst = cv2.applyColorMap(img, 12)
# cv2.imwrite('poisson_noise.jpg', dst)
cv2.imshow('map', dst)
cv2.imwrite('./scatter_rgb_airplane/airplane1.png',dst,[cv2.IMWRITE_PNG_COMPRESSION, 0])
# cv2.waitKey()
# cv2.destroyAllWindows()
