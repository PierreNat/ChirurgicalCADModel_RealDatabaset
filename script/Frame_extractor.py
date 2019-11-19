import cv2
import random
import os
print(cv2.__version__)
vidcap = cv2.VideoCapture('data/IFBS_ENDOSCOPE.mp4')
success,image = vidcap.read()
count = 0
item = random.sample(range(20000), 19228)
print(item[0])
success = True

# while success:
#   var = item[count]
#   cv2.imwrite("framesLeft_random/frameL%d.jpg" % var, image[0:1024,0:1280,:])     # save frame as JPEG file
#   success,image = vidcap.read()
#   print('Read a new frame: ', success)
#   count += 1
#   print(count)

# #rename file
# i = 0
#
# for filename in os.listdir("framesLeft_random/"):
#   dst = "frameL" + str(i) + ".jpg"
#   src = 'framesLeft_random/' + filename
#   dst = 'framesLeft/' + dst
#
#   # rename() function will
#   # rename all the files
#   os.rename(src, dst)
#   i += 1
#   print(i)