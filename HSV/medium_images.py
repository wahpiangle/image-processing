import matplotlib.pyplot as plt
import cv2
import numpy as np
from helper import imageMap
from easy_images import threshold_image_hsv, threshold_image_with_blur


medium_map:dict = [
    {
        "image": "medium1",
        "lower": [30, 90, 10],
        "upper": [50, 100, 100]
    },
    {
        "image": "medium2",
        "lower": [15, 5, 23],
        "upper": [55, 100, 100]
    },
    {
        "image": "medium3",
        "lower": [35, 15, 60],
        "upper": [65, 100, 100]
    }
]

plt.figure(figsize=(20, 20))

for i in range(len(medium_map)):
    plt.subplot(4, 3, i+1)
    plt.imshow(cv2.cvtColor(threshold_image_hsv(imageMap[medium_map[i]["image"]], medium_map[i]["lower"], medium_map[i]["upper"]), cv2.COLOR_RGB2BGR))
    plt.axis('off')
    plt.title(medium_map[i]["image"])

for i in range(len(medium_map)):
    plt.subplot(4, 3, i+4)
    plt.imshow(cv2.cvtColor(threshold_image_with_blur(imageMap[medium_map[i]["image"]], medium_map[i]["lower"], medium_map[i]["upper"], 5), cv2.COLOR_RGB2BGR))
    plt.axis('off')
    plt.title(medium_map[i]["image"] + " with blur")