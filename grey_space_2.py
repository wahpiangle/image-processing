import cv2
import numpy as np
import matplotlib.pyplot as plt

input_path = "Dataset/input_images/"
output_path = "output_images/"

easy1_image = cv2.imread(input_path + "easy/easy_1.jpg")
easy2_image = cv2.imread(input_path + "easy/easy_2.jpg")
easy3_image = cv2.imread(input_path + "easy/easy_3.jpg")
medium1_image = cv2.imread(input_path + "medium/medium_1.jpg")
medium2_image = cv2.imread(input_path + "medium/medium_2.jpg")
medium3_image = cv2.imread(input_path + "medium/medium_3.jpg")
hard1_image = cv2.imread(input_path + "hard/hard_1.jpg")
hard2_image = cv2.imread(input_path + "hard/hard_2.jpg")
hard3_image = cv2.imread(input_path + "hard/hard_3.jpg")
imageMap: dict = {
    "easy1": easy1_image,
    "easy2": easy2_image,
    "easy3": easy3_image,
    "medium1": medium1_image,
    "medium2": medium2_image,
    "medium3": medium3_image,
    "hard1": hard1_image,
    "hard2": hard2_image,
    "hard3": hard3_image
}


def apply_gamma_correction(image, gamma):
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # Apply the gamma correction using the lookup table
    return cv2.LUT(image, table)

# edge sharpening
def sharpen_image_laplacian(image):
    # Apply Laplacian filter for edge detection
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    # Add the Laplacian image back to the original image to sharpen the edges
    sharpened_image = np.uint8(image + laplacian)
    return sharpened_image

def sharpen_image_kernel(image):
    # Define the sharpening kernel
    kernel = np.array([ [-1, -1, -1],
                        [-1,  9, -1],
                        [-1, -1, -1]])
    
    # Apply the filter using cv2.filter2D()
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image


final_result = []

# Load each image, convert to grayscale, and apply gamma correction
for key, image in imageMap.items():
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    gamma_corrected_image = apply_gamma_correction(gray_image, gamma=1.5)
    
    sharpened_image = sharpen_image_kernel(gamma_corrected_image)
    
    filtered_image = cv2.bilateralFilter(sharpened_image, d=9, sigmaColor=120, sigmaSpace=120)
    
    # otsu 
    #_, thresholded_image = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    
    # watershed
    filtered_image = np.uint8(filtered_image)
    _, thresh = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Plot the thresholded image
    plt.figure(figsize=(6, 6))
    plt.imshow(thresh, cmap='gray')
    plt.title(f'{key} - Thresholded (OTSU)')
    plt.axis('off')
    plt.show()
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    segmented_image = cv2.watershed(image, markers)
    
    final_result.append(segmented_image)
    


plt.figure(figsize=(12, 8))

for i, image in enumerate(final_result, 1):
    plt.subplot(3, 3, i)  # Adjust based on the number of images
    plt.imshow(image, cmap='gray')
    plt.title(f'{i}')
    plt.axis('off')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()