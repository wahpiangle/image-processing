import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from helper import imageMap, normalize_hsv, pipeline_path, convert_to_binary


def threshold_image_hsv(image, lower, upper, lower1, upper1, index):
    image = cv2.medianBlur(image, 21)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array(normalize_hsv(lower[0], lower[1], lower[2]))
    upper = np.array(normalize_hsv(upper[0], upper[1], upper[2]))
    mask = cv2.inRange(hsv, lower, upper)

    lower1 = np.array(normalize_hsv(lower1[0], lower1[1], lower1[2]))
    upper1 = np.array(normalize_hsv(upper1[0], upper1[1], upper1[2]))
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask = mask + mask1
    final_image = cv2.bitwise_and(image, image, mask=mask)
    final_image = cv2.GaussianBlur(final_image, (21, 21), 5)
    final_image = convert_to_binary(final_image, 40)
    path = os.path.join(pipeline_path, "medium", "trial_and_error")
    os.path.exists(path) or os.makedirs(path)
    cv2.imwrite(os.path.join(path, f"{index}.jpg"), final_image)
    return final_image


medium_map: dict = [
    {
        "image": "medium1",
        "lower": [30, 50, 10],
        "upper": [55, 100, 100],
        "lower1": [0, 0, 10],
        "upper1": [360, 20, 100],
    },
    {
        "image": "medium2",
        "lower": [15, 5, 20],
        "upper": [55, 100, 100],
        "lower1": [0, 0, 40],
        "upper1": [360, 30, 100],
    },
    {
        "image": "medium3",
        "lower": [35, 15, 20],
        "upper": [70, 100, 100],
        "lower1": [0, 0, 20],
        "upper1": [360, 20, 100],
    },
]

plt.figure(figsize=(20, 20))

for i in range(len(medium_map)):
    plt.subplot(4, 3, i + 1)
    plt.imshow(
        cv2.cvtColor(
            threshold_image_hsv(
                imageMap[medium_map[i]["image"]],
                medium_map[i]["lower"],
                medium_map[i]["upper"],
                medium_map[i]["lower1"],
                medium_map[i]["upper1"],
                i,
            ),
            cv2.COLOR_RGB2BGR,
        )
    )
    plt.axis("off")
    plt.title(medium_map[i]["image"])

# for i in range(len(medium_map)):
#     plt.subplot(4, 3, i + 4)
#     plt.imshow(
#         cv2.cvtColor(
#             threshold_image_with_blur(
#                 imageMap[medium_map[i]["image"]],
#                 medium_map[i]["lower"],
#                 medium_map[i]["upper"],
#                 5,
#             ),
#             cv2.COLOR_RGB2BGR,
#         )
#     )
#     plt.axis("off")
#     plt.title(medium_map[i]["image"] + " with blur")
