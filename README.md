# Attendance-check---face-matching

Importing necessary libraries:

The code begins by importing required libraries such as DeepFace, pandas, argparse, cv2 (OpenCV), numpy, and os.
Setting image paths and database path:

img1_path and img2_path variables store the file paths of two images (1.jpg and 2.jpeg).
db_path stores the directory path containing a database of images.
Command-line argument parsing:

The code uses the argparse library to handle command-line arguments for different functionalities.
It allows users to specify different options like --detection, --verify, --similar, --embeddings, --analyze, or --similarity to perform various face analysis tasks.
Face Detection and Alignment (args.detection):

If the --detection argument is provided, the code detects faces in the first image (img1_path) using various face detection backends (e.g., retinaface, dlib, etc.).
Bounding boxes are drawn around the detected faces with confidence percentages displayed.
The image with bounding boxes is displayed using OpenCV's cv2.imshow().
Face Verification (args.verify):

If the --verify argument is provided, the code performs face verification between the two images (img1_path and img2_path).
The verification result (whether the faces belong to the same person or not) is printed.
Face Similarity (args.similar):

If the --similar argument is provided, the code finds similar faces in the database (db_path) to the first image (img1_path).
The similarity score between the detected faces and the database faces is calculated (using cosine similarity by default) and printed.
The top similar face images are displayed side by side using OpenCV's cv2.imshow().
Face Embeddings (args.embeddings):

If the --embeddings argument is provided, the code computes the face embeddings (vector representations) of the faces in the first image (img1_path) using the "Facenet" model.
The embeddings are then printed.
Face Attributes Analysis (args.analyze):

If the --analyze argument is provided, the code analyzes the facial attributes (age, gender, race, emotion) of the faces in the first image (img1_path).
The age, dominant gender, dominant race, and dominant emotion are extracted and displayed on the image using OpenCV.
Face Attributes Analysis for Multiple Images in a Folder (folder_path):

If a folder_path is provided (commented out in this case), the code iterates through all images in the specified folder.
For each image, it analyzes the facial attributes and stores the results in a list (filename_results).
The list is then printed, containing information about age, gender, race, and emotion for each image.
Face Similarity with Different Metrics (args.similarity):

If the --similarity argument is provided, the code performs face verification and face recognition using different distance metrics (cosine, euclidean, euclidean_l2).
The verification result (whether the faces belong to the same person or not) and recognition result (similar faces from the database) are printed.
Note: To execute specific functionalities, you can provide the corresponding command-line arguments while running the script. For example, to perform face detection, run the script with the --detection flag: python script_name.py --detection.
