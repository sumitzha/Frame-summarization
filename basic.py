import cv2
import numpy as np
from datetime import timedelta
from ffmpy import FFmpeg
import os
from matplotlib import pyplot as plt

# Global Variables
total_frames_processed = 0
# Set a threshold for motion detection
threshold = 60


# Function to detect motion in a specific ROI
def detect_motion(roi_gray, current_roi_gray, frame, bg_subtractor):

    blurred = cv2.GaussianBlur(current_roi_gray, (5, 5), 1)
    # cv2.imshow("Gaussian_Blur",blurred)
   
    fg_mask = bg_subtractor.apply(blurred)
    # cv2.imshow("Background_Subtraction",fg_mask)

    _, thresholded = cv2.threshold(fg_mask, threshold, 255, cv2.THRESH_BINARY)
    # cv2.imshow("Thresolded_Frame",thresholded)

    kernel = np.ones((5, 5), np.uint8)
    closed_thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("After_MORPH_OPEN",closed_thresholded)

    contours, _ = cv2.findContours(closed_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    # Filter contours based on minimum area (300 in this case)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 300]

    # Draw rectangles around detected contours
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return len(contours) > 0


def main(vid_file,roi):

    global total_frames_processed  

    # Open a video capture object (0 is usually the default camera)
    cap = cv2.VideoCapture(vid_file)

    # Check if the video capture is successfully opened
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        exit()

    # Define video settings (adjust as needed)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames_orig = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        
    # Create VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # You can change the codec as needed
    out = cv2.VideoWriter('static\processed\og.mp4', fourcc, fps, (width, height))

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read the first frame.")
        exit()

    # Define multiple Regions of Interest (ROIs) coordinates
    rois = []
    if len(roi) == 4:
        rois = [ (roi) ]
    else:
        rois = [(0, 0, width, height)]

    print(roi)

    # Convert the ROIs to grayscale for motion detection
    rois_gray = [cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY) for x, y, w, h in rois]

    # Initialize motion tracking array
    motion_detected = [0] * total_frames_orig

    # # Create MOG background subtractor
    # bg_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG2(
    #     history=500, nmixtures=5, backgroundRatio=0.7, noiseSigma=0
    # )

    # Create MOG background subtractor
    history = 500

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=150,
        detectShadows=False
    )


    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Check if motion is detected in any of the ROIs
        motion_detected[frame_number - 1] = any(
            detect_motion(rois_gray[i], cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY), frame, bg_subtractor)
            for i, (x, y, w, h) in enumerate(rois)
        )

        # Save the frame to the output video if motion is detected
        if motion_detected[frame_number - 1]:
            # Calculate the timestamp based on the frame number and frame rate
            timestamp_seconds = frame_number / fps
            timestamp = str(timedelta(seconds=timestamp_seconds))

            # Draw the timestamp on the frame
            cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            out.write(frame)
            cv2.imshow('Motion Detection', frame)
        
            total_frames_processed += 1

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture object, VideoWriter, and close the window
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Print the motion_detected array
    # print("Motion Detection Array:", motion_detected)
    # print("fps", fps)
    print(total_frames_processed, total_frames_orig, (total_frames_processed/total_frames_orig)*100)

    # ff = FFmpeg(inputs={'static/video/output/og.mp4': None}, outputs={'static/video/output/output.mp4':'-c:v h264 -c:a ac3'})
    # ff.run()
    # os.remove("static/video/output/og.mp4")

