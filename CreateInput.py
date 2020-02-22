import FormatData
import cv2
import numpy as np
import h5py
import os


img = cv2.imread("testimages/equation1.png")

img = FormatData.black_and_white(img)
img = FormatData.crop_image(img)
characters = FormatData.extract_characters(img)

images = []
labels = []
for char in characters:
    images.append(char)
    labels.append(0)


