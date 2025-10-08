import cv2
import numpy as np

# Load original image
img = cv2.imread('../Rock block.tif')  # Path to your original image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Resize image to 875 x 875
resized_img = cv2.resize(gray, (875, 875))
cv2.imwrite('../Rock_block_resized.jpg', resized_img)

# --- Edge Detection ---

# 1. Canny
edges_canny = cv2.Canny(resized_img, 100, 200)
cv2.imwrite('../edges_canny.jpg', edges_canny)

# 2. Laplacian of Gaussian (LoG)
blurred = cv2.GaussianBlur(resized_img, (5,5), 0)
edges_log = cv2.Laplacian(blurred, cv2.CV_64F)
edges_log = cv2.convertScaleAbs(edges_log)
cv2.imwrite('../edges_log.jpg', edges_log)

# 3. Sobel
sobelx = cv2.Sobel(resized_img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(resized_img, cv2.CV_64F, 0, 1, ksize=3)
edges_sobel = cv2.magnitude(sobelx, sobely)
edges_sobel = cv2.convertScaleAbs(edges_sobel)
cv2.imwrite('../edges_sobel.jpg', edges_sobel)

# 4. Prewitt
kernelx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
edges_prewitt_x = cv2.filter2D(resized_img, -1, kernelx)
edges_prewitt_y = cv2.filter2D(resized_img, -1, kernely)
edges_prewitt = cv2.magnitude(edges_prewitt_x.astype(np.float32), edges_prewitt_y.astype(np.float32))
edges_prewitt = cv2.convertScaleAbs(edges_prewitt)
cv2.imwrite('../edges_prewitt.jpg', edges_prewitt)

# 5. Roberts Cross
kernelx = np.array([[1,0],[0,-1]])
kernely = np.array([[0,1],[-1,0]])
edges_roberts_x = cv2.filter2D(resized_img, -1, kernelx)
edges_roberts_y = cv2.filter2D(resized_img, -1, kernely)
edges_roberts = cv2.magnitude(edges_roberts_x.astype(np.float32), edges_roberts_y.astype(np.float32))
edges_roberts = cv2.convertScaleAbs(edges_roberts)
cv2.imwrite('../edges_roberts.jpg', edges_roberts)

print("Resizing and edge detection complete!")
