# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 14:01:46 2023

@author: sb
"""

import os
import cv2
import imutils


def draw_color_mask(img, borders, color=(0, 0, 0)):
    h = img.shape[0]
    w = img.shape[1]

    x_min = int(borders[0] * w / 100)
    x_max = w - int(borders[2] * w / 100)
    y_min = int(borders[1] * h / 100)
    y_max = h - int(borders[3] * h / 100)

    img = cv2.rectangle(img, (0, 0), (x_min, h), color, -1)
    img = cv2.rectangle(img, (0, 0), (w, y_min), color, -1)
    img = cv2.rectangle(img, (x_max, 0), (w, h), color, -1)
    img = cv2.rectangle(img, (0, y_max), (w, h), color, -1)

    return img


def preprocess_image_change_detection(img, gaussian_blur_radius_list=None, black_mask=(5, 10, 5, 0)):
    gray = img.copy()
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gaussian_blur_radius_list is not None:
        for radius in gaussian_blur_radius_list:
            gray = cv2.GaussianBlur(gray, (radius, radius), 0)

    gray = draw_color_mask(gray, black_mask)

    return gray


def compare_frames_change_detection(prev_frame, next_frame, min_contour_area):
    frame_delta = cv2.absdiff(prev_frame, next_frame)
    thresh = cv2.threshold(frame_delta, 45, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh, None, iterations=2)
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    score = 0
    res_cnts = []
    for c in cnts:
        
        if cv2.contourArea(c) < min_contour_area:
            continue
        print('Cnt_area: ',cv2.contourArea(c))
        res_cnts.append(c)
        score += cv2.contourArea(c)
    print('Score: ',score)
    return score, res_cnts, thresh


def are_images_similar_size(img1, img2, max_size_difference=0):
    height1, width1 = img1.shape
    height2, width2 = img2.shape
    size_difference = abs(width1 - width2) + abs(height1 - height2)
    if not size_difference <= max_size_difference:
        print('Image sizes are different. Skipping.')
    return size_difference <= max_size_difference


def is_valid_image(img):
    # Check if the image is valid (not empty and has proper shape)
    return img is not None and img.shape[0] > 0 and img.shape[1] > 0


def find_and_remove_similar_images(input_folder):
    min_contour_area_threshold = 3000
    min_score_threshold = 100000
    min_count_threshold = 25000
    max_size_difference = 0

    image_files = [file for file in os.listdir(input_folder) if file.endswith('.png')]
    print(image_files)
    num_images = len(image_files)

    images_to_remove = []

    for i in range(num_images):
        img_path_i = os.path.join(input_folder, image_files[i])
        img_i = cv2.imread(img_path_i)

        if not is_valid_image(img_i):
            print(f"Warning: Image {img_path_i} is invalid or corrupted. Skipping.")
            images_to_remove.append(image_files[i])
            continue

        img_i = preprocess_image_change_detection(img_i)

        for j in range(i + 1, num_images):
            img_path_j = os.path.join(input_folder, image_files[j])
            img_prefix_i = image_files[i][:3]
            img_prefix_j = image_files[j][:3]

            # Check if the first 3 letters of the image file names match
            if img_prefix_i != img_prefix_j:
                print('Images are from two different Camera IDs. Skipping')
                continue

            img_j = cv2.imread(img_path_j)
            print(f"Comparing Image:{image_files[i]} with Image:{image_files[j]}")

            if not is_valid_image(img_j):
                print(f"Warning: Image {img_path_j} is invalid or corrupted. Skipping.")
                images_to_remove.append(image_files[j])
                continue

            img_j = preprocess_image_change_detection(img_j)

            if are_images_similar_size(img_i, img_j, max_size_difference):
                score, _, thresh = compare_frames_change_detection(img_i, img_j, min_contour_area_threshold)

                count = 0
                height, width = thresh.shape
                for y in range(height):
                    for x in range(width):
                        # Check pixel value at (x, y)
                        pixel_value = thresh[y, x]

                        # Check if the pixel has changed (difference in intensity)
                        if pixel_value == 255:
                            count = count + 1
                print('Count: ', count)

                if score < min_score_threshold and count < min_count_threshold:
                    # Images are similar, add image_files[j] to the images_to_remove list
                    images_to_remove.append(image_files[j])

    # Remove images that need to be removed from the original image_files list
    #image_files = [file for file in image_files if file not in images_to_remove]

    # Perform the removal of similar images from the dataset folder
    for image_to_remove in images_to_remove:
        img_path_to_remove = os.path.join(input_folder, image_to_remove)
        print(f"Removing similar image: {img_path_to_remove}")
        os.remove(img_path_to_remove)


if __name__ == "__main__":
    input_folder = r"C:\Users\sb\Downloads\dataset-candidates-ml_final\dataset"
    find_and_remove_similar_images(input_folder)
