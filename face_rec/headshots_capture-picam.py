import cv2
import os
from datetime import datetime
import time
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Capture headshots with camera')
parser.add_argument('--pie-cam', '--picam', action='store_true', 
                    help='Use Raspberry Pi camera (default: False, uses webcam)')
args = parser.parse_args()
pie_cam = args.pie_cam

if pie_cam:
    from picamera2 import Picamera2

# Change this to the name of the person you're photographing
PERSON_NAME = "Shefeeq2"

def create_folder(name):
    dataset_folder = "dataset"
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    
    person_folder = os.path.join(dataset_folder, name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
    return person_folder

def capture_photos(name):
    folder = create_folder(name)
    if pie_cam:
        # Initialize the camera
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
        picam2.start()
    else:
        # OpenCV webcam capture
        cap = cv2.VideoCapture(0)  # 0 = default webcam

    # Allow camera to warm up
    time.sleep(2)

    photo_count = 0
    
    print(f"Taking photos for {name}. Press SPACE to capture, 'q' to quit.")
    
    while True:
        if pie_cam:
            # Capture frame from Pi Camera
            frame = picam2.capture_array()
        else:
            ret, frame = cap.read()

        
        # Display the frame
        cv2.imshow('Capture', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space key
            photo_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.jpg"
            filepath = os.path.join(folder, filename)
            cv2.imwrite(filepath, frame)
            print(f"Photo {photo_count} saved: {filepath}")
        
        elif key == ord('q'):  # Q key
            break
    
    # Clean up
    cv2.destroyAllWindows()
    if pie_cam:
        picam2.stop()
    else:
        cap.release()
    print(f"Photo capture completed. {photo_count} photos saved for {name}.")

if __name__ == "__main__":
    capture_photos(PERSON_NAME)