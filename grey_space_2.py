import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

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
    "hard3": hard3_image,
}


def apply_gamma_correction(image, gamma):
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype(
        "uint8"
    )
    # Apply the gamma correction using the lookup table
    return cv2.LUT(image, table)


# edge sharpening
def sharpen_image_laplacian(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    sharpened_image = np.uint8(image + laplacian)
    return sharpened_image


def sharpen_image_kernel(image):
    # Define the sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

# Function to create a directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to save images in the specified directory
def save_image(image, directory, filename):
    cv2.imwrite(os.path.join(directory, filename), image)

# Create directories for each step
output_directories = ["gamma_corrected", "sharpened", "filtered", "thresholded", "morphological", "segmented"]
for directory in output_directories:
    create_directory(os.path.join(output_path, directory))

final_result = []

for key, image in imageMap.items():
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gamma_corrected_image = apply_gamma_correction(gray_image, gamma=1.5)
    save_image(gamma_corrected_image, os.path.join(output_path, "gamma_corrected"), f"{key}.jpg")

    sharpened_image = sharpen_image_kernel(gamma_corrected_image)
    save_image(sharpened_image, os.path.join(output_path, "sharpened"), f"{key}.jpg")

    filtered_image = cv2.bilateralFilter(sharpened_image, d=11, sigmaColor=120, sigmaSpace=120)
    save_image(filtered_image, os.path.join(output_path, "filtered"), f"{key}.jpg")

    _, thresh = cv2.threshold(
        filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    save_image(thresh, os.path.join(output_path, "thresholded"), f"{key}.jpg")

    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=18)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    save_image(opening, os.path.join(output_path, "morphological"), f"{key}.jpg")

    segmented_image = cv2.watershed(image, markers)
    segmented_image[segmented_image == 1] = 0
    segmented_image[segmented_image != 0] = 255

    save_image(segmented_image, os.path.join(output_path, "segmented"), f"{key}.jpg")

    # Ensure correct data type
    segmented_image = np.uint8(segmented_image)
    # get the binary image
    segmented_image = cv2.bitwise_and(image, image, mask=segmented_image)
    final_result.append(segmented_image)

plt.figure(figsize=(12, 8))

for i, image in enumerate(final_result, 1):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.subplot(3, 3, i)
    plt.imshow(rgb)
    plt.title(f"{i}")
    plt.axis("off")

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()
