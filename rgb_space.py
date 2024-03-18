import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

input_path = "Dataset/input_images/"
output_path = "output_images/"

easy1_image = cv.imread(input_path + "easy/easy_1.jpg")
easy2_image = cv.imread(input_path + "easy/easy_2.jpg")
easy3_image = cv.imread(input_path + "easy/easy_3.jpg")
medium1_image = cv.imread(input_path + "medium/medium_1.jpg")
medium2_image = cv.imread(input_path + "medium/medium_2.jpg")
medium3_image = cv.imread(input_path + "medium/medium_3.jpg")
hard1_image = cv.imread(input_path + "hard/hard_1.jpg")
hard2_image = cv.imread(input_path + "hard/hard_2.jpg")
hard3_image = cv.imread(input_path + "hard/hard_3.jpg")
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

def thresholding():
    root = os.getcwd()
    
    for i in range(1, 4):
        img_filename = f"medium_{i}.jpg"
        imgPath = os.path.join(root, f"Dataset\\input_images\\medium\\{img_filename}")
        img = cv.imread(imgPath)

        # checks for bgr (default color space for opencv)
        if img.shape[2] == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        imgGray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        hist = cv.calcHist([imgGray], [0], None, [256], [0, 256])
        plt.figure()
        plt.plot(hist)
        plt.xlabel("Bins")
        plt.ylabel("# of pixels")
        plt.title(f"Histogram of {img_filename}")
        plt.show()

        
        threshOptions = [
            cv.THRESH_BINARY,
            cv.THRESH_BINARY_INV,
            cv.THRESH_TRUNC,
            cv.THRESH_TOZERO,
            cv.THRESH_TOZERO_INV,
        ]

        threshNames = ["BINARY", "BINARY_INV", "TRUNC", "TOZERO", "TOZERO_INV"]

        plt.figure(figsize=(20, 10))
        plt.subplot(231)
        plt.imshow(imgGray, cmap="gray")

        for j in range(len(threshOptions)):
            plt.subplot(2, 3, j + 2)
            _, imgThresh = cv.threshold(imgGray, 70, 225, threshOptions[j])
            plt.imshow(imgThresh, cmap="gray")
            plt.title(threshNames[j])

        plt.show()

if __name__ == "__main__":
    thresholding()