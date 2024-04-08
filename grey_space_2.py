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
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image


final_result = []

for key, image in imageMap.items():
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gamma_corrected_image = apply_gamma_correction(gray_image, gamma=1.5)
    sharpened_image = sharpen_image_kernel(gamma_corrected_image)

    filtered_image = cv2.bilateralFilter(
        sharpened_image, d=9, sigmaColor=120, sigmaSpace=120
    )

    # # otsu
    # filtered_image = np.uint8(filtered_image)
    _, thresh = cv2.threshold(
        filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Plot the thresholded image
    # plt.figure(figsize=(6, 6))
    # plt.imshow(thresh, cmap="gray")
    # plt.title(f"{key} - Thresholded (OTSU)")
    # plt.axis("off")
    # plt.show()
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=0)
    sure_bg = cv2.dilate(opening, kernel, iterations=9)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.55 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    segmented_image = cv2.watershed(image, markers)
    segmented_image[segmented_image == 1] = 0
    segmented_image[segmented_image != 0] = 255

    final_result.append(markers)

plt.figure(figsize=(12, 8))

for i, image in enumerate(final_result, 1):
    plt.subplot(3, 3, i)
    plt.imshow(image, cmap="gray")
    plt.title(f"{i}")
    plt.axis("off")

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()
