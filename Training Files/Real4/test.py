import cv2
import time

# Try different indices if 1 does not work. This assumes the external camera is at index 1.
cap = cv2.VideoCapture(1)
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)  # Create a resizable window
cv2.resizeWindow('Frame', 640, 480)  # Set the window size


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()