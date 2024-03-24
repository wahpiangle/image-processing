import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from helper import normalize_hsv, imageMap, HSVCHANNEL, pipeline_path, convert_to_binary
from skimage.color import rgb2hsv


def threshold_image_hsv(image, lower, upper, index):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array(normalize_hsv(lower[0], lower[1], lower[2]))
    upper = np.array(normalize_hsv(upper[0], upper[1], upper[2]))
    mask = cv2.inRange(hsv, lower, upper)
    final_image = cv2.bitwise_and(image, image, mask=mask)
    final_image = convert_to_binary(final_image)
    path = os.path.join(pipeline_path, "easy", "trial_and_error")
    os.path.exists(path) or os.makedirs(path)
    cv2.imwrite(os.path.join(path, f"{index}.jpg"), final_image)
    return final_image


def threshold_image_with_blur(image, lower, upper, blur, index):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array(normalize_hsv(lower[0], lower[1], lower[2]))
    upper = np.array(normalize_hsv(upper[0], upper[1], upper[2]))
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    final_image = cv2.bitwise_and(image, image, mask=mask)
    final_image = convert_to_binary(final_image)
    path = os.path.join(pipeline_path, "easy", "trial_and_error")
    cv2.imwrite(os.path.join(path, f"gaussianBlur_size{blur}_{index}.jpg"), final_image)
    return final_image


def threshold_by_hsv_channel(image, lower, upper, channel: HSVCHANNEL, index, gaussian):
    channel_title = channel.name
    if gaussian:
        image = cv2.GaussianBlur(image, (gaussian, gaussian), 0)
    sample_h = rgb2hsv(image)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    if channel != HSVCHANNEL.VALUE:
        im = ax[0].imshow(sample_h[:, :, 0], cmap="hsv")
        fig.colorbar(im, ax=ax[0])

    ax[0].imshow(sample_h[:, :, channel.value], cmap="hsv")
    ax[0].set_title(channel_title, fontsize=15)
    ax[0].axis("off")

    ax[1].imshow(sample_h[:, :, 1], cmap="hsv")
    ax[1].set_title("Saturation", fontsize=15)
    ax[1].axis("off")

    ax[2].imshow(sample_h[:, :, 2], cmap="hsv")
    ax[2].set_title("Value", fontsize=15)
    ax[2].axis("off")

    lower_mask = sample_h[:, :, channel.value] > lower
    upper_mask = sample_h[:, :, channel.value] < upper

    mask = upper_mask * lower_mask
    red = image[:, :, 0] * mask
    green = image[:, :, 1] * mask
    blue = image[:, :, 2] * mask
    final_image = np.dstack((red, green, blue))
    ax[1].imshow(mask)

    # convert final image to BGR
    path = os.path.join(pipeline_path, "easy", "hsv_channel_segmentation")
    os.path.exists(path) or os.makedirs(path)
    cv2.imwrite(os.path.join(path, f"{index}.jpg"), final_image)
    ax[2].imshow(final_image)
    ax[1].set_title("Mask", fontsize=15)
    plt.show()
    return


plt.figure(figsize=(15, 15))
plt.axis("off")
# plt.title("Trial & Error Method")
easy_map: dict = [
    {"image": "easy1", "lower": [45, 10, 50], "upper": [70, 100, 100]},
    {"image": "easy2", "lower": [45, 51, 50], "upper": [70, 100, 100]},
    {"image": "easy3", "lower": [45, 0, 82], "upper": [66, 100, 100]},
]

easy_map_hsv: dict = [
    {"image": "easy1", "lower": 0.47, "upper": 0.55, "channel": HSVCHANNEL.HUE},
    {"image": "easy2", "lower": 0.48, "upper": 0.53, "channel": HSVCHANNEL.HUE},
    {
        "image": "easy3",
        "lower": 0.85,
        "upper": 1.5,
        "channel": HSVCHANNEL.VALUE,
        "gaussian": 35,
    },
]
# Method 1: Trial and error method
for i in range(len(easy_map)):
    plt.subplot(4, 3, i + 1)
    plt.imshow(
        cv2.cvtColor(
            threshold_image_hsv(
                imageMap[easy_map[i]["image"]],
                easy_map[i]["lower"],
                easy_map[i]["upper"],
                i,
            ),
            cv2.COLOR_RGB2BGR,
        )
    )
    plt.axis("off")
    plt.title(easy_map[i]["image"])

for i in range(len(easy_map)):
    plt.subplot(4, 3, i + 4)
    plt.imshow(
        cv2.cvtColor(
            threshold_image_with_blur(
                imageMap[easy_map[i]["image"]],
                easy_map[i]["lower"],
                easy_map[i]["upper"],
                5,
                i,
            ),
            cv2.COLOR_RGB2BGR,
        )
    )
    plt.axis("off")
    plt.title(easy_map[i]["image"] + " (With Gaussian Blur of kernel size 5)")

for i in range(len(easy_map)):
    plt.subplot(4, 3, i + 7)
    plt.imshow(
        cv2.cvtColor(
            threshold_image_with_blur(
                imageMap[easy_map[i]["image"]],
                easy_map[i]["lower"],
                easy_map[i]["upper"],
                11,
                i,
            ),
            cv2.COLOR_RGB2BGR,
        )
    )
    plt.axis("off")
    plt.title(easy_map[i]["image"] + " (Gaussian Blur of kernel size 11)")

# Method 2 HSV Channel Segmentation
for i in range(len(easy_map_hsv)):
    threshold_by_hsv_channel(
        imageMap[easy_map_hsv[i]["image"]],
        easy_map_hsv[i]["lower"],
        easy_map_hsv[i]["upper"],
        easy_map_hsv[i]["channel"],
        i,
        easy_map_hsv[i].get("gaussian", None),
    )
