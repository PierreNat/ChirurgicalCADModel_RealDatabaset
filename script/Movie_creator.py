import cv2
import os
import tqdm
image_folder = 'framesLeft'
video_name = 'framesleftvideo.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 60, (width,height))
span = 0
for image in tqdm.tqdm(images):
    if span%10 == 0:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    span = span+1

cv2.destroyAllWindows()
video.release()

# import cv2
# import numpy as np
# import os
# from os.path import isfile, join
#
# pathIn = 'framesgif/cameraSet1/'
# pathOut = 'videotest.avi'
# fps = 60
# frame_array = []
# files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
# # for sorting the file names properly
# files.sort(key=lambda x: x[5:-4])
# files.sort()
# frame_array = []
# files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
# # for sorting the file names properly
# files.sort(key=lambda x: x[5:-4])
# for i in range(len(files)):
#     filename = pathIn + files[i]
#     print(files[i])
#     # reading each files
#     img = cv2.imread(filename)
#     height, width, layers = img.shape
#     size = (width, height)
#
#     # inserting the frames into an image array
#     frame_array.append(img)
# out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'XVID'), fps, size)
# for i in range(len(frame_array)):
#     # writing to a image array
#     out.write(frame_array[i])
# out.release()