import cv2
from matplotlib import pyplot as plt


def firstFrame(vid_file):
    # Open a video capture object (0 is usually the default camera)
    cap = cv2.VideoCapture(vid_file)

    # Check if the video capture is successfully opened
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        exit()

    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Display the image using pyplot
    plt.imshow(rgb_frame)
    plt.title('First Frame')
    plt.show()


# Put path here to know better about co-ordinates
firstFrame(r"sample.mp4")


