import os

import cv2
import numpy as np
import screeninfo

screen_id = 0

# get the size of the screen
screen = screeninfo.get_monitors()[screen_id]
width, height = screen.width, screen.height
image = np.ones((height, width, 3), dtype=np.float32)
window_name = "projector"
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

key_wait_time = 25
video_path = "./checkpoints/fire_project/output/"
output_path = "/home/indo/code/pix2pix_video/checkpoints/fire_project/output"
org_video = "brain.mov"
terminate = False
fps = 30
intervals = list(range(0,2001,100))
trans_videos = list(
    map(lambda x: os.path.join(output_path, x), sorted(os.listdir(output_path)))
)

while True:
    frame_idx = 0
    interval_idx = 0
    is_interact = False
    trans_cap = None
    interact_frame = -1
    cap = cv2.VideoCapture(org_video)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx == intervals[-1]:
            break
        elif frame_idx == intervals[interval_idx + 1]:
            interval_idx += 1
            if is_interact and trans_cap is not None:
                is_interact = False
                trans_cap = None

        if is_interact:
            print(frame_idx, intervals[interval_idx], trans_cap)
            if intervals[interval_idx] == frame_idx:
                current_video = trans_videos[interval_idx]
                trans_cap = cv2.VideoCapture(current_video)
                ret, trans_frame = trans_cap.read()
                cv2.imshow(window_name, trans_frame)
            elif (
                trans_cap is not None
                and intervals[interval_idx]
                < frame_idx
                < intervals[interval_idx+1]
            ):
                ret, trans_frame = trans_cap.read()
                cv2.imshow(window_name, trans_frame)
            else:
                cv2.imshow(window_name, frame)

        else:
            cv2.imshow(window_name, frame)

        frame_idx += 1

        # Press Q on keyboard to exit
        if cv2.waitKey(key_wait_time) & 0xFF == ord("q"):
            terminate = True
            break

        # detect interaction
        if cv2.waitKey(key_wait_time) & 0xFF == ord("i"):
            print("Interact!")
            is_interact = True

    if terminate:
        break
    cap.release()
cv2.destroyAllWindows()
