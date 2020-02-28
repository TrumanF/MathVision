import cv2
import numpy as np
import math
import h5py
import GenerateData
import Keras
import re
import sympy
from sympy.parsing.sympy_parser import parse_expr
from sympy.utilities.lambdify import lambdify, implemented_function
from sympy.abc import x
from sympy.parsing.sympy_parser import standard_transformations,implicit_multiplication_application

f = h5py.File("C:/Users/Truman/Documents/GitHub/MathVision/HDF5/f16_characters_test.h5", "r")

dset_labels = f.get("labels").value
dset_images = f.get("images").value
dset_expo = [x[1] for x in dset_labels]

predicts = Keras.make_predictions(dset_images)

print(predicts)
print(dset_expo)


def join_characters(characters, expo_lst):
    alpha_lst = []
    edited_characters = characters.copy()
    for char, i in zip(characters, range(len(characters))):
        # Check if variable
        if char.isalpha():
            # Check if variable is already in list
            if char not in alpha_lst:
                alpha_lst.append(char)
        # Set carat to python exponent operator
        if char == "^":
            edited_characters[i] = "**"

        if expo_lst[i] == 1:
            if expo_lst[i-1] == 1:
                pass
            else:
                edited_characters[i] = '**' + characters[i]
        if char == "(":
            edited_characters[i] = '*' + characters[i]

    final_string = ''.join(edited_characters)
    if "=" in final_string:
        lst = final_string.split("=")
        final_string = lst[0] + '-(' + lst[1] + ')'

    transformations = (standard_transformations +
                       (implicit_multiplication_application, ))
    a = parse_expr(final_string, transformations=transformations)
    print(sympy.solve(a, x))

join_characters(predicts, dset_expo)

