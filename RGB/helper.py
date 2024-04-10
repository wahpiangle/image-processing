import cv2 as cv
import os

# Get the absolute path to the directory containing the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define relative paths to input and output directories
input_rel_path = "../Dataset/input_images/"
output_rel_path = "../output_images/"

# Create absolute paths by joining the current directory with the relative paths
input_abs_path = os.path.join(current_dir, input_rel_path)
output_abs_path = os.path.join(current_dir, output_rel_path)

# Load images using absolute paths
easy1_image = cv.imread(os.path.join(input_abs_path, "easy/easy_1.jpg"))
easy2_image = cv.imread(os.path.join(input_abs_path, "easy/easy_2.jpg"))
easy3_image = cv.imread(os.path.join(input_abs_path, "easy/easy_3.jpg"))
medium1_image = cv.imread(os.path.join(input_abs_path, "medium/medium_1.jpg"))
medium2_image = cv.imread(os.path.join(input_abs_path, "medium/medium_2.jpg"))
medium3_image = cv.imread(os.path.join(input_abs_path, "medium/medium_3.jpg"))
hard1_image = cv.imread(os.path.join(input_abs_path, "hard/hard_1.jpg"))
hard2_image = cv.imread(os.path.join(input_abs_path, "hard/hard_2.jpg"))
hard3_image = cv.imread(os.path.join(input_abs_path, "hard/hard_3.jpg"))

# Create image map with absolute paths
imageMap = {
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
