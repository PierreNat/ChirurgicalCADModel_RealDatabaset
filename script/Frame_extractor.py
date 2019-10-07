import cv2

print(cv2.__version__)
vidcap = cv2.VideoCapture('data/IFBS_ENDOSCOPE.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
  cv2.imwrite("framesLeft/frameL%d.jpg" % count, image[0:1024,0:1280,:])     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1