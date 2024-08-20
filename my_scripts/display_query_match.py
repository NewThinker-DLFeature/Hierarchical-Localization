#!/usr/bin/env python3

import cv2
import numpy as np

def read_image_pairs(file_path, name_filter):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    pairs = [line.strip().split() for line in lines if name_filter in line ]
    return pairs

def display_images(image_pairs, dataset):
    index = 0
    total_pairs = len(image_pairs)
    
    while True:
        img1_path, img2_path = image_pairs[index]
        img1_path = dataset + "/query/" + img1_path
        img2_path = dataset + "/database/" + img2_path
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            print(f"Error loading images: {img1_path}, {img2_path}")
            break

        # Resize images to the same height
        height = min(img1.shape[0], img2.shape[0])
        img1 = cv2.resize(img1, (int(img1.shape[1] * height / img1.shape[0]), height))
        img2 = cv2.resize(img2, (int(img2.shape[1] * height / img2.shape[0]), height))

        # Concatenate images horizontally
        combined_image = np.hstack((img1, img2))

        cv2.imshow('Image Comparison', combined_image)

        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC key
            break
        elif key == 81:  # Left arrow key
            index = (index - 1) % total_pairs
        elif key == 83:  # Right arrow key
            index = (index + 1) % total_pairs

    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys

    #file_path = 'image_pairs.txt'
    file_path = sys.argv[1]
    dataset = sys.argv[2]
    if len(sys.argv) > 3:
        name_filter = sys.argv[3]
    else:
        name_filter = "_c0_"


    image_pairs = read_image_pairs(file_path, name_filter)
    display_images(image_pairs, dataset)


