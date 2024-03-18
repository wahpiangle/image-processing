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
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # mean filtering
    blurred = cv2.blur(gray, (5,5))
    
    # bilateral filtering
    # blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    return blurred

def segment_flower_manual(image, threshold_value, max_value, threshold_type):
    # Manual thresholding to segment flower material
    _, thresh = cv2.threshold(image, threshold_value, max_value, threshold_type)
    
    return thresh

def post_process(binary_image):
    # Morphological operations for noise removal and smoothing
    kernel = np.ones((5,5),np.uint8)

    opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    # erosion = cv2.erode(binary_image,kernel,iterations = 3)
    dilation = cv2.dilate(closed,kernel,iterations = 5)
    
    return dilation 

def process_image(image):
    # Preprocessing
    preprocessed = preprocess_image(image)
    
    # Segmentation   last parameter = threshold type
    segmented = segment_flower_manual(preprocessed, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Post-processing
    processed = post_process(segmented)
    
    return processed


fig, axs = plt.subplots(3, 3, figsize=(15, 15))

# Process and plot each image in the imageMap dictionary
for i, (name, img) in enumerate(imageMap.items()):
    processed_img = process_image(img)
    row = i // 3
    col = i % 3
    axs[row, col].imshow(processed_img, cmap='gray')
    axs[row, col].set_title(name + " Processed")
    
plt.tight_layout()
plt.show()