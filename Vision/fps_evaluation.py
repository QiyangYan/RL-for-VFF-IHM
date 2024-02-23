import cv2
import time

def test_fps():
    # Initialize the camera capture
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Failed to open the camera.")
        return

    # Variables for FPS calculation
    frames_to_skip = 30  # Number of initial frames to skip for stabilization
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Skip initial frames for stabilization
            if frame_count < frames_to_skip:
                frame_count += 1
                continue

            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time

            # Display FPS
            print(f"FPS: {fps:.2f}")
            height, width, _ = frame.shape
            print(f"RGB Resolution: {width}x{height}")

            # Display the frame
            # cv2.imshow("Frame", frame)

            # Check for the 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Release the camera
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_fps()