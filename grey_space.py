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



def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Gaussian blur
    blurred = cv2.GaussianBlur(gray, (43,43 ), 0)
    
    # mean filtering
    #blurred = cv2.blur(gray, (25,25))
    
    # bilateral filtering
    #blurred = cv2.bilateralFilter(gray, 9, 255, 255)
    
    return blurred

def segment_flower_manual(image, threshold_value, max_value, threshold_type):
    # Manual thresholding to segment flower material
    _, thresh = cv2.threshold(image, threshold_value, max_value, threshold_type)
    
    return thresh

def post_process(binary_image):
    # Morphological operations for noise removal and smoothing
    kernel = np.ones((5,5),np.uint8)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_rectangular = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    
    
    # erosion -> dilation -> close  BETTER 
    erosion = cv2.erode(binary_image, kernel_ellipse, iterations=5)
    open = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, np.ones((5,5),np.uint8),iterations=3)
    dilation = cv2.dilate(open, kernel, iterations=7)
    dilation_2 = cv2.dilate(dilation, kernel_ellipse, iterations=3)
    closed = cv2.morphologyEx(dilation_2, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8), iterations=2)
    
    
    
    return closed

def process_image(image):
    # Preprocessing
    preprocessed = preprocess_image(image)
    
    
    
    # Segmentation   last parameter = threshold type
    segmented = segment_flower_manual(preprocessed, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #segmented = segment_flower_manual(edges, 100,255, cv2.THRESH_BINARY)
    
    
    # Post-processing
    processed = post_process(segmented)
    # edge detection
    #edges = cv2.Canny(processed, 50,50,3)
    return processed


fig, axs = plt.subplots(3, 3, figsize=(15, 15))

# Process and plot each image in the imageMap dictionary
for i, (name, img) in enumerate(imageMap.items()):
    processed_img = process_image(img)
    row = i // 3
    col = i % 3
    axs[row, col].imshow(processed_img, cmap='gray')
    
plt.tight_layout()
plt.show()