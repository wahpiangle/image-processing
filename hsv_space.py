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

def normalize_hsv(h:int , s:int, v:int): 
    return [h/2, s/100*255, v/100*255]

def threshold_image_hsv(image, lower, upper):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array(normalize_hsv(lower[0], lower[1], lower[2]))
    upper = np.array(normalize_hsv(upper[0], upper[1], upper[2]))
    mask = cv2.inRange(hsv, lower, upper)
    return cv2.bitwise_and(image, image, mask=mask)

def threshold_image_with_blur(image, lower, upper, blur):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array(normalize_hsv(lower[0], lower[1], lower[2]))
    upper = np.array(normalize_hsv(upper[0], upper[1], upper[2]))
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    return cv2.bitwise_and(image, image, mask=mask)

plt.figure(figsize=(20, 20))

easy_map: dict = [
    {
        "image": "easy1",
        "lower": [45, 0, 50],
        "upper": [70, 100, 100]
    },
    {
        "image": "easy2",
        "lower": [45, 40, 50],
        "upper": [70, 100, 100]
    },
    {
        "image": "easy3",
        "lower": [45, 0, 83],
        "upper": [70, 100, 100]
    },
]

for i in range(len(easy_map)):
    plt.subplot(4, 3, i+1)
    plt.imshow(cv2.cvtColor(threshold_image(imageMap[easy_map[i]["image"]], easy_map[i]["lower"], easy_map[i]["upper"]), cv2.COLOR_RGB2BGR))
    plt.axis('off')
    plt.title(easy_map[i]["image"])

plt.subplot(4, 3, 4)
plt.imshow(cv2.cvtColor(threshold_image_with_blur(imageMap["easy1"], [45, 0, 50], [70, 100, 100], 5), cv2.COLOR_RGB2BGR))
plt.axis('off')
plt.title("easy1 (With Gaussian Blur)")

plt.subplot(4, 3, 5)
plt.imshow(cv2.cvtColor(threshold_image_with_blur(imageMap["easy2"], [45, 40, 50], [70, 100, 100], 5), cv2.COLOR_RGB2BGR))
plt.axis('off')
plt.title("easy2 (With Gaussian Blur)")

plt.subplot(4, 3, 6)
plt.imshow(cv2.cvtColor(threshold_image_with_blur(imageMap["easy3"], [45, 0, 82], [66, 100, 100], 5), cv2.COLOR_RGB2BGR))
plt.axis('off')
plt.title("easy3 (With Gaussian Blur)")
