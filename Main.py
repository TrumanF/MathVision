"""
Write code to open any file in testimages file with .png extension
"""


import pytesseract
import cv2
import numpy as np
import os
import math
import h5py

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract'


# eq_no is the number of the test file
def main(eq_no, user_input):
    eq_no += 1
    img_dict = {}
    for i in range(1, eq_no):
        img_dict["img{0}".format(i)] = cv2.imread('testimages/individual{0}_test.png'.format("2"))
    """
    For each image opened do 5 things:
    1) Resize image (optional)
    2) Blur the image with GaussianBlur
    3) Convert image to grayscale
    4) Convert image to black and white
    5) Convert to inverted black and white
    Then set original image to new, modified image
    """
    for image in img_dict:
        scale = 1
        blur = (3, 3)
        temp_img = img_dict[image]
        temp_img = cv2.resize(temp_img, (round(temp_img.shape[1] * scale), round(temp_img.shape[0] * scale)))
        temp_img_blur = cv2.GaussianBlur(temp_img, blur, 0)
        temp_img_gray = cv2.cvtColor(temp_img_blur, cv2.COLOR_BGR2GRAY)
        (thresh, temp_img_baw) = cv2.threshold(temp_img_gray, 127, 255, cv2.THRESH_BINARY)
        temp_img_inv = cv2.bitwise_not(temp_img_baw)
        img_dict[image] = temp_img_inv

    # Get contours of characters in image
    # Iterate through each contour and draw a bounding rectangle
    # Create 'boxes' list to store rectangles created
    def find_contours(input_img):
        """ Finds contours of input image using cv2.findContours(). Returns boxes created.
        Parameters:
        ---------------
        image   input image to find contours for
        """
        contours, hierarchy = cv2.findContours(input_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        boxes = np.zeros((len(contours), 4))
        j = 0
        for c in contours:
            j += 1
            added = False
            (x, y, w, h) = cv2.boundingRect(c)
            for i in range(len(boxes)):
                # Check if there is another rectangle that is somewhat above it (within 20 pixels) and add them as the
                # same rectangle
                try:
                    if boxes[i][0] - 5 <= x <= boxes[i][0] + 5 and boxes[i][1] - 15 <= y <= boxes[i][1] + 15:
                        temp_boxes = np.delete(boxes, boxes[i])
                        boxes[i] = [min(x, boxes[i][0]), min(y, boxes[i][1]), max(x + h, boxes[i][2]), max(y + h, boxes[i][3])]
                        boxes.resize((boxes.shape[0]-1, 4))
                        added = True
                        j -= 1
                        break
                except IndexError:
                    continue

            if not added:
                boxes[j-1] = [x, y, x + w, y + h]
        return boxes

    def crop_image(input_img, boxes):
        """ Crops image based off of generated boxes. Returns cropped image.
        Note: Currently does not work on skewed images
        Parameters:
        ---------------
        input_img       input image to crop
        boxes           boxes of all found contours
        """
        sz = 3
        left = int(np.min(boxes[:, 0])) - sz
        top = int(np.min(boxes[:, 1])) - sz
        right = int(np.max(boxes[:, 2])) + sz
        bottom = int(np.max(boxes[:, 3])) + sz
        temp_image = input_img[top:bottom, left:right]
        # Call find_contours again because contour locations have now changed
        return temp_image

    def draw_boxes(input_img, boxes):
        """ Draws rectangles on input image, given the box data of that image. Returns new modified image.
        Parameters:
        ---------------
        boxes       array of rectangular data: (x, y, w+x, h+y)     i.e. cv2.findContours() output
        image       input image to draw rectangles on
        """
        temp_img = input_img
        for box in boxes:
            temp_box = [int(element) for element in box]
            temp_img = cv2.rectangle(temp_img, (temp_box[0], temp_box[1]), (temp_box[2], temp_box[3]), 255, 1)
        return temp_img

    def create_boxes(boxes, image, name):
        """ Creates boxes for given input image and writes them to a .box file.
        Parameters:
        ---------------
        boxes       array of rectangular data: (x, y, w+x, h+y)     i.e. cv2.findContours() output
        image       input image to create boxes for
        name        name of .box file to be created
        """
        f = open("{0}{1}.box".format(name, eq_no), "w+")
        for box in boxes:
            temp_box = [int(element) for element in box]
            temp_box = (temp_box[0], int(image.shape[0]) - temp_box[1], temp_box[2], int(image.shape[0]) - temp_box[3])
            f.write(("x " + str(temp_box).strip("[]()") + " 0" + "\n").replace(",", ""))
        f.close()

    # Set directory for final images to be saved
    directory = r'C:\Users\Truman\Documents\GitHub\MathVision\testimages\trainingdata'
    os.chdir(directory)

    boxes_dict = {}
    for image in img_dict:
        # Find boxes of each character
        boxes = find_contours(img_dict[image])
        # Crop image
        img_dict[image] = crop_image(img_dict[image], boxes)
        # Find new boxes for new cropped image
        boxes_dict[image] = find_contours(img_dict[image])

    characters_dict = {}
    for image in img_dict:
        i = 0
        for box in boxes_dict[image]:
            i += 1
            # Set a safezone of 3 pixels around the character
            sz = 3
            # Create temp_box list that converts all elements in boxes to integers (from floats).
            temp_box = [int(element) for element in box]
            characters_dict["character{0}_{1}".format(i, image)] = img_dict[image][temp_box[1] - sz:temp_box[3] + sz, temp_box[0] - sz:temp_box[2] + sz]

    # Find character with largest size in both x and y
    max_x = 0
    max_y = 0
    for character in characters_dict:
        if characters_dict[character].shape[0] > max_y:
            max_y = characters_dict[character].shape[0]
        if characters_dict[character].shape[1] > max_x:
            max_x = characters_dict[character].shape[1]

    # Resize all characters to be same size as character with largest dimension by adding black space to all sides

    # For this to work with current keras model, the size has to be identical between characters, so size must be set
    # to a constant, like:
    # size = 32

    size = max(max_x, max_y)
    if size % 2 != 0:
        size += 1
    for image in characters_dict:
        extra_y = 0
        extra_x = 0
        if characters_dict[image].shape[0] % 2 == 1:
            extra_y = 1
        if characters_dict[image].shape[1] % 2 == 1:
            extra_x = 1
        b_y = int((size - characters_dict[image].shape[0])/2)
        b_x = int((size - characters_dict[image].shape[1])/2)
        characters_dict[image] = cv2.copyMakeBorder(characters_dict[image], b_y + extra_y, b_y, b_x + extra_x, b_x, 1, (0, 0, 0))

    length = math.floor(len(characters_dict) / 10)
    remainder = len(characters_dict) % 10
    blank_img = np.zeros((1, size, size), dtype="uint8")

    # Create grid of all characters found in original input image
    final_concat_dict = {}
    resized_chars_copy = np.asarray(list(characters_dict.values()))

    character_row = []
    # Make rows of 10 characters
    for i in range(length):
        h_concat = cv2.hconcat(resized_chars_copy[10*i:10*(i+1)])
        # Check if this is the last row to create
        if i == length - 1 and remainder != 0:
            # Add blank sections for end of photo
            for j in range(10 - remainder):
                resized_chars_copy = np.append(resized_chars_copy, blank_img, axis=0)
            h_concat_last = cv2.hconcat(resized_chars_copy[10*(i+1):10*(i+2)])
            character_row.append(h_concat)
            character_row.append(h_concat_last)
            break
        character_row.append(h_concat)
    # Produce final grids with 10 items in each row and column
    k = -1
    for row in range(1, len(character_row)+1):
        if row % 10 == 0:
            k += 1
            final_concat_dict["final_concat_{}".format(k)] = cv2.vconcat([character_row[x] for x in range(k*10, (k+1)*10)])
    # If there are extra characters, put them all into the final image, which may be less than 10 rows
    final_concat_dict["final_concat_{}".format(k)] = cv2.vconcat([character_row[x] for x in range(k*10, (k+1)*10)])

    final_img_list = []
    for image in characters_dict:
        final_img_list.append(characters_dict[image])

    for image in final_concat_dict:
        cv2.imshow("image", final_concat_dict[image])
        cv2.waitKey()
    print(size)
    return final_img_list


def store_many_hdf5(images, labels):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)
    if user_input == "train":
        hdf5_dir = "C:/Users/Truman/Documents/GitHub/MathVision/HDF5/f{0}_characters_y.h5".format(num_images)
    if user_input == "test":
        hdf5_dir = "C:/Users/Truman/Documents/GitHub/MathVision/HDF5/f{0}_characters_test_y.h5".format(num_images)
    # Create a new HDF5 file
    file = h5py.File(hdf5_dir, "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
    )
    meta_set = file.create_dataset(
        "meta", np.shape(labels), h5py.h5t.STD_U8BE, data=labels
    )
    file.close()
    return hdf5_dir


if __name__ == "__main__":
    user_input = input("Would you like to train or test?").lower()
    characters = main(1, user_input)
    path = store_many_hdf5(characters, characters)
    print("Complete! New HDF5 stored at {0}".format(path))

