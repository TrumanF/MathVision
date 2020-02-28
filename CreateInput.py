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
    cv2.imshow("char", char[0])
    cv2.waitKey()
    char = FormatData.resize_characters(char, size)
    images.append(char[0])
    labels.append([0, char[2]])
# Temporary label generator; Delete later

# x2+6(7x+10y)=z13
for char, i in zip([x for x in "6+7+21"], range(len(labels))):
    number = GenerateData.characters.index(char)
    labels[i][0] = number

print(labels)
FormatData.store_many_hdf5(images, labels, "frominput")
