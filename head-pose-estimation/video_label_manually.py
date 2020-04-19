import cv2
import numpy as np

cap = cv2.VideoCapture('output_video.avi')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output_classes.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5, (frame_width, frame_height))

print(frame_width, frame_height)

current_state = 0

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (20, 50)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 1

while(True):
    # Read one frame.
    ret, frame = cap.read()
    if not ret:
        break

    # Show one frame.
    cv2.imshow('frame', frame)

    # Check, if the space bar is pressed to switch the mode.
    k = cv2.waitKey(0)
    text = ""
    if k == ord('1'):
        text = "Attentive"
        print(text)
    elif k == ord('2'):
        text = "Distracted"
        print(text)
    elif k == ord('3'):
        text = "Uses phone"
        print(text)
    elif k == ord('4'):
        text = "Slightly inattentive"
        print(text)
    elif k == ord('5'):
        text = "Tired"
        print(text)
    elif k == ord('6'):
        text = "Absent"
        print(text)
    cv2.putText(frame, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    out.write(frame)

out.release()
cap.release()
cv2.destroyAllWindows()