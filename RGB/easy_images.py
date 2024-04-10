import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from helper import imageMap


def unsharp_masking(img, sigma=1.0, strength=1.5):
    blurred = cv.GaussianBlur(img, (0, 0), sigma)
    sharpened = cv.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return sharpened

def rgb_thresholding(image_key):
    img = imageMap[image_key]
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Apply Gaussian Blur for noise reduction
    noiseReduction = cv.GaussianBlur(imgRGB, (5, 5), 0)

    easy_thresholds_rgb = {
        "easy1": {
            "lower_threshold": np.array([170, 170, 0]),
            "upper_threshold": np.array([255, 255, 255]),
        },
        "easy2": {
            "lower_threshold": np.array([140, 140, 0]),
            "upper_threshold": np.array([255, 255, 255]),
        },
        "easy3": {
            "lower_threshold": np.array([190, 190, 0]),
            "upper_threshold": np.array([255, 255, 255]),
        },
    }

    # Get the threshold values for the current image key
    threshold_values = easy_thresholds_rgb.get(image_key)

    # Check if threshold values exist for the current image key
    if threshold_values is not None:
        lower_threshold = threshold_values["lower_threshold"]
        upper_threshold = threshold_values["upper_threshold"]
    else:
        # If no specific threshold values are defined, use default values
        lower_threshold = np.array([170, 170, 0])  # Lower bound of the threshold range
        upper_threshold = np.array(
            [255, 255, 255]
        )  # Upper bound of the threshold range

    # Threshold the blurred image based on the RGB color range
    mask_blurred = cv.inRange(noiseReduction, lower_threshold, upper_threshold)
    masked_blurred_img = cv.bitwise_and(
        noiseReduction, noiseReduction, mask=mask_blurred
    )

    # Convert the masked image to grayscale
    masked_gray_img = cv.cvtColor(masked_blurred_img, cv.COLOR_RGB2GRAY)

    # Apply binary thresholding
    _, binary_img = cv.threshold(masked_gray_img, 0, 255, cv.THRESH_BINARY)

    # Displaying the original image, noise-reduced image, and thresholded blurred image
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(imgRGB)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(noiseReduction)
    plt.title("Noise Reduction")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(masked_blurred_img)
    plt.title("Blurred Image Thresholding")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(masked_gray_img, cmap='gray')
    plt.title("Black and White Masked Image")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(binary_img, cmap='gray')
    plt.title("Binary Thresholding")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    for key in imageMap:
        # Check if the image is "easy" based on some criteria
        if key.startswith("easy"):
            rgb_thresholding(key)
