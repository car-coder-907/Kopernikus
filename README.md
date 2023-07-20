# Kopernikus
Programming challenge for Perception team member in Kopernikus

The are_images_similar_size function checks if the size of two images is similar within a given max_size_difference. The is_valid_image function checks whether the image is valid, meaning it is not empty and has proper dimensions. The main function find_and_remove_similar_images takes the input folder path as an argument. It starts by defining the min_contour_area_threshold, min_score_threshold, and max_size_difference parameters. The function reads the image files in the input folder and iterates through each image.

For each image, it checks if the image is valid and then applies the preprocess_image_change_detection function to convert the image to a grayscale version with a black mask. It then iterates through the remaining images with the same prefix (first 3 letters of the image file name) to compare them with the current image. If the images have the same prefix, it checks if they are similar in size based on the are_images_similar_size function.

If the images are similar in size, it proceeds to compare them using the compare_frames_change_detection function. The contour areas are calculated for both images, and if the score (sum of contour areas) is below the min_score_threshold, the second image is considered similar to the first one.If a similar image is found, it is removed from the input folder using os.remove, and the filename of the removed image is appended to the removeimages list, along with information about the similarity. The script continues this process for each image in the folder, comparing it with all other images that share the same prefix (belong to the same camera ID group).

The script aims to identify and remove similar-looking images from the dataset, ensuring that only unique and diverse images remain within each camera ID group. It does so by comparing images based on contour area similarity.
