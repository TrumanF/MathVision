import FormatData
import cv2
import numpy as np
import h5py
import os
import GenerateData

img = cv2.imread("testimages/equation1.png")

img = FormatData.black_and_white(img)
img = FormatData.crop_image(img)
characters = FormatData.extract_characters(img)

images = []
labels = []
size = FormatData.max_character_size(characters)

for char in characters:
    char = FormatData.resize_characters(char, size)
    cv2.imshow("char", char[0])
    cv2.waitKey()
    images.append(char[0])
for char in "=zyx++01761)32(":
    number = GenerateData.characters.index(char)
    labels.append(number)
FormatData.store_many_hdf5(images, labels, "test")
