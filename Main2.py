import pytesseract
import cv2
import numpy as np
import os
import math

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract'


def main(eq_no):
    # Open Image
    img = cv2.imread('testimages/equation{0}.png'.format(eq_no))
    # Resize image to 3X
    img = cv2.resize(img, (round(img.shape[1] * 2), round(img.shape[0] * 2)))
    # Blur image
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    # Make image gray scale, then black and white
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    # Invert black and white image
    img_inv = cv2.bitwise_not(blackAndWhiteImage)

    # Get contours of characters in image
    # Iterate through each contour and draw a bounding rectangle
    # Create 'boxes' list to store rectangles created
    def find_contours(input_img):
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
                    if boxes[i][0] - 10 <= x <= boxes[i][0] + 10 and boxes[i][1] - 20 <= y <= boxes[i][1] + 20:
                        temp_boxes = np.delete(boxes, boxes[i])
                        print("Removing", boxes[i], "because of", (x, y, w, h))
                        boxes[i] = [min(x, boxes[i][0]), min(y, boxes[i][1]), max(x + h, boxes[i][2]), max(y + h, boxes[i][3])]
                        boxes.resize((boxes.shape[0]-1, 4))
                        print("New size", boxes.shape[0])
                        added = True
                        j -= 1
                        break
                except IndexError:
                    continue

            if not added:
                boxes[j-1] = [x, y, x + w, y + h]
        return boxes

    def crop_image(input_img):
        nonlocal boxes
        sz = 5
        left = int(np.min(boxes[:, 0])) - sz
        top = int(np.min(boxes[:, 1])) - sz
        right = int(np.max(boxes[:, 2])) + sz
        bottom = int(np.max(boxes[:, 3])) + sz
        temp_image = input_img[top:bottom, left:right]
        # Call find_contours again because contour locations have now changed
        return temp_image

    def draw_boxes(input_img, boxes):
        temp_img = input_img
        for box in boxes:
            temp_box = [int(element) for element in box]
            temp_img = cv2.rectangle(temp_img, (temp_box[0], temp_box[1]), (temp_box[2], temp_box[3]), 255, 1)
        return temp_img

    boxes = find_contours(img_inv)
    img_cropped = crop_image(img_inv)
    boxes = find_contours(img_cropped)
    # img_boxes = draw_boxes(img_cropped, boxes)

    # Set directory for final images to be saved
    directory = r'C:\Users\Truman\Documents\GitHub\MathVision\testimages\trainingdata'
    os.chdir(directory)

    characters = []

    for box in boxes:
        sz = 3
        temp_box = [int(element) for element in box]
        characters.append(img_cropped[temp_box[1]-sz:temp_box[3]+sz, temp_box[0]-sz:temp_box[2]+sz])


    # Resize all characters to be same size by adding black space to all sides
    resized_chars = []
    for image in characters:
        size = 90
        extra_y = 0
        extra_x = 0
        if image.shape[0] % 2 == 1:
            extra_y = 1
        if image.shape[1] % 2 == 1:
            extra_x = 1
        b_y = int((size - image.shape[0])/2)
        b_x = int((size - image.shape[1])/2)
        image = cv2.copyMakeBorder(image, b_y + extra_y, b_y, b_x + extra_x, b_x, 1, (0, 0, 0))
        resized_chars.append(image)



    # i = -1
    # for image in characters:
    #     i += 1
    #     scale = 1
    #     border = 5
    #     image = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
    #     image = cv2.copyMakeBorder(image, border, border, border, border, 1, (0, 0, 0))
    #     #cv2.imshow("Image2", image)
    #     #text = pytesseract.image_to_string(image, config='--psm 10')
    #     #print(text)
    #     #cv2.imwrite("mat.cambriamath.exp{0}.png".format(i), image)

    # Create grid of all characters found in original input image
    character_row = []
    length = math.floor(len(resized_chars) / 10)
    print(length)
    remainder = len(resized_chars) % 10
    print(remainder)
    blank_img = np.zeros((90, 90), dtype="uint8")

    # Make rows of 10 characters
    for i in range(length):
        h_concat = cv2.hconcat(resized_chars[10*i:10*(i+1)])
        # Check if this is the last row to create
        if i == length - 1 and remainder != 0:
            # Add blank sections for end of photo
            for j in range(10 - remainder):
                print("adding blank", i)
                resized_chars.append(blank_img)
            h_concat_last = cv2.hconcat(resized_chars[10*(i+1):10*(i+2)])
            character_row.append(h_concat)
            character_row.append(h_concat_last)
            break
        character_row.append(h_concat)


    # Take all rows and vertically concatenate into one large grid
    final_concat = cv2.vconcat([character_row[x] for x in range(len(character_row))])
    boxes_characters = find_contours(final_concat)
    # final_concat_draw = draw_boxes(final_concat, boxes_characters)

    f = open("mat.cambriamath.exp{0}.box".format(eq_no), "w+")
    for box in boxes_characters:
        temp_box = [int(element) for element in box]
        temp_box = (temp_box[0], int(final_concat.shape[0]) - temp_box[1], temp_box[2], int(final_concat.shape[0]) - temp_box[3])
        f.write(("x " + str(temp_box).strip("[]()") + " 0" + "\n").replace(",", ""))
    f.close()

    img_final = img_cropped
    text1 = pytesseract.image_to_string(resized_chars[-9], config="--psm 10 -l mat+eng")
    cv2.imshow("Character", resized_chars[-10])
    text2 = pytesseract.image_to_string(img_final, config="-l mat+eng")
    print(text1)
    print(text2)
    cv2.imshow("Image3", final_concat)
    cv2.imshow("Image", img_final)
    #cv2.imwrite("mat.cambriamath.exp{0}.png".format(eq_no), final_concat)
    cv2.waitKey()


if __name__ == "__main__":
        main(1)


