import cv2
import numpy as np

# Global variables to store points
src_points = []
dst_points = []

def select_points(image, points, window_title):
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(window_title, image)

    cv2.namedWindow(window_title)
    cv2.setMouseCallback(window_title, mouse_callback)
    cv2.imshow(window_title, image)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or len(points) >= 4:  # Change 4 to the number of points you need
            break

    cv2.destroyWindow(window_title)

def main(video_path):
    # Load a frame from the video
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to grab a frame from the video")
        return

    # Select src_points in the camera view
    select_points(frame, src_points, "Select Source Points (Press 'q' when done)")

    # Create an empty canvas for dst_points selection
    canvas_size = (640, 480)  # Adjust the size as needed
    canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)

    # Select dst_points on the canvas
    select_points(canvas, dst_points, "Select Destination Points (Press 'q' when done)")

    print("Source Points:", src_points)
    print("Destination Points:", dst_points)

if __name__ == "__main__":
    video_path = "/Users/adonaivera/Documents/ai-model-dot/input/cam5Parallel_1.mp4"  # Update this to your video path
    main(video_path)