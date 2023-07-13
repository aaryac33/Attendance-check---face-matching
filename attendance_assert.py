import pandas as pd
import cv2
import datetime
import os

from deepface import DeepFace

# Set the paths and variables
db_path = r"C:/Users/aarya/OneDrive/Desktop/face recog/aarya_2.jpg"  # Path to the database of employee images
attendance_folder = r"C:/Users/aarya/OneDrive/Desktop/face recog/attendance"  # Folder to save attendance records
attendance_records = []  # List to store attendance records

# Define a function to process frames
def process_frame(frame):
    # Perform face verification
    result = DeepFace.verify(img1_path=frame, img2_path=db_path, enforce_detection=False)

    # Check if a face is detected and recognized with sufficient confidence
    if result["verified"]:
        identity = result["verified"]
        confidence = result["distance"]
        if confidence < 0.5:  # Adjust the confidence threshold as needed
            # Retrieve employee information from the database
            employee_name = identity  # Replace with actual method to retrieve name
            employee_age = 30  # Replace with actual method to retrieve age
            employee_gender = "Male"  # Replace with actual method to retrieve gender

            # Check if the person is entering or exiting
            if len(attendance_records) > 0 and attendance_records[-1]["name"] == employee_name:
                # Person is exiting
                exit_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                attendance_records[-1]["exit_time"] = exit_time
            else:
                # Person is entering
                entry_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                attendance_records.append({
                    "name": employee_name,
                    "age": employee_age,
                    "gender": employee_gender,
                    "entry_time": entry_time,
                    "exit_time": None
                })

# Set up video capture
cap = cv2.VideoCapture(0)  # Adjust the camera index if needed

# Start the loop for processing frames
while True:
    ret, frame = cap.read()  # Read a frame from the camera

    # Perform the necessary operations on the frame
    process_frame(frame)

    # Display the frame if needed
    cv2.imshow("Camera", frame)

    # Check for key press to exit the loop
    if cv2.waitKey(1) == ord("q"):  # Press 'q' to exit
        break

# Save the attendance records to a CSV file
os.makedirs(attendance_folder, exist_ok=True)
attendance_path = os.path.join(attendance_folder, "attendance.csv")
df = pd.DataFrame(attendance_records)
df.to_csv(attendance_path, index=False)

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
