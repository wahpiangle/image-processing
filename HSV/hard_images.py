import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from helper import imageMap, normalize_hsv, pipeline_path, convert_to_binary


def threshold_image_hsv(
    image,
    lower,
    upper,
    lower1,
    upper1,
    index,
    gaussian,
    lower2,
    upper2,
    median_blur,
):
    if gaussian:
        image = cv2.GaussianBlur(image, (gaussian, gaussian), 5)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array(normalize_hsv(lower[0], lower[1], lower[2]))
    upper = np.array(normalize_hsv(upper[0], upper[1], upper[2]))
    mask = cv2.inRange(hsv, lower, upper)

    lower1 = np.array(normalize_hsv(lower1[0], lower1[1], lower1[2]))
    upper1 = np.array(normalize_hsv(upper1[0], upper1[1], upper1[2]))
    lower2 = np.array(normalize_hsv(lower2[0], lower2[1], lower2[2]))
    upper2 = np.array(normalize_hsv(upper2[0], upper2[1], upper2[2]))

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = mask + mask1 + mask2
    final_image = cv2.bitwise_and(image, image, mask=mask)
    if median_blur:
        final_image = cv2.medianBlur(final_image, 21)
    # final_image = convert_to_binary(final_image, median_blur)
    path = os.path.join(pipeline_path, "hard", "trial_and_error")
    os.path.exists(path) or os.makedirs(path)
    final_image = convert_to_binary(final_image)
    cv2.imwrite(os.path.join(path, f"{index}.jpg"), final_image)
    return final_image


hard_map: dict = [
    {
        "image": "hard1",
        "lower": [35, 20, 45],
        "upper": [65, 100, 100],
        "lower1": [0, 0, 40],
        "upper1": [360, 20, 100],
        "gaussian": 11,
    },
    {
        "image": "hard2",
        "lower": [30, 20, 30],
        "upper": [69, 100, 100],
        "lower1": [0, 0, 30],
        "upper1": [360, 28, 100],
        "gaussian": 21,
        "median_blur": 21,
    },
    {
        "image": "hard3",
        "lower": [35, 20, 45],
        "upper": [65, 100, 100],
        "lower1": [0, 0, 40],
        "upper1": [360, 20, 100],
        "gaussian": 11,
    },
]

plt.figure(figsize=(20, 20))

for i in range(len(hard_map)):
    plt.subplot(4, 3, i + 1)
    plt.imshow(
        cv2.cvtColor(
            threshold_image_hsv(
                image=imageMap[hard_map[i]["image"]],
                lower=hard_map[i]["lower"],
                upper=hard_map[i]["upper"],
                lower1=hard_map[i]["lower1"],
                upper1=hard_map[i]["upper1"],
                index=i,
                gaussian=hard_map[i]["gaussian"] if "gaussian" in hard_map[i] else 0,
                lower2=hard_map[i]["lower2"] if "lower2" in hard_map[i] else [0, 0, 0],
                upper2=hard_map[i]["upper2"] if "upper2" in hard_map[i] else [0, 0, 0],
                median_blur=(
                    hard_map[i]["median_blur"] if "median_blur" in hard_map[i] else 0
                ),
            ),
            cv2.COLOR_RGB2BGR,
        )
    )
    plt.axis("off")
    plt.title(hard_map[i]["image"])
