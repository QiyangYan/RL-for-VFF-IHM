import cv2

cap = cv2.VideoCapture(1)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow('Img', cv2.WINDOW_NORMAL)  # Create a resizable window
cv2.resizeWindow('Img', 640, 480)  # Set the window size

num = 0

while cap.isOpened():

    succes, img = cap.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('images/img' + str(num) + '.png', img)
        print("image saved!")
        num += 1

    cv2.imshow('Img', img)

# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()