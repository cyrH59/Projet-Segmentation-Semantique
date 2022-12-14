import cv2
vidcap = cv2.VideoCapture('video.mp4')
success,image = vidcap.read()
count = 0
while success:
    cv2.imwrite("../images/frames/%d.png" % count, image)     # save frame as JPEG file
    success,image = vidcap.read()
    count += 1
