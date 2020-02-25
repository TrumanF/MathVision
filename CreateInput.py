import FormatData
import cv2
import numpy as np
import h5py
import os
import GenerateData

# TODO: Add label generation for input image.

img = cv2.imread("testimages/arithmetic1.png")

img = FormatData.black_and_white(img)
img = FormatData.crop_image(img)
characters = FormatData.extract_characters(img)

images = []
labels = []
size = FormatData.max_character_size(characters)

for char in characters:
    char = FormatData.resize_characters(char, size)
    cv2.imshow(str(char[1]), char[0])
    cv2.waitKey()
    images.append(char[0])
# Temporary label generator for equation1
for char in "6+7+21":
    number = GenerateData.characters.index(char)
    labels.append(number)

FormatData.store_many_hdf5(images, labels, "test")
