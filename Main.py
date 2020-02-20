import nltk
import pytesseract
import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract'
image = cv2.imread('testimages/trainingdata/eng.cambriamath.exp1.png')

image = cv2.resize(image, (round(image.shape[1] * 3), round(image.shape[0] * 3)))

image_blur = cv2.GaussianBlur(image, (3, 3), 0)

grayImage = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
image_inv = cv2.bitwise_not(blackAndWhiteImage)
new_image = image_inv
boxes = []


def find_contours():
    contours, hierarchy = cv2.findContours(new_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    for c in contours:
        added = False
        (x, y, w, h) = cv2.boundingRect(c)
        for box in boxes:
            if box[0] - 10 <= x <= box[0] + 10:
                boxes.remove(box)
                boxes.append([min(x, box[0]), min(y, box[1]), max(x + h, box[2]), max(y + h, box[3])])
                added = True
                break
        if not added:
            boxes.append([x, y, x+w, y+h])


find_contours()
boxes = np.asarray(boxes)
print(boxes)
print(len(boxes))

# need an extra "min/max" for contours outside the frame
left = np.min(boxes[:, 0])
top = np.min(boxes[:, 1])
right = np.max(boxes[:, 2])
bottom = np.max(boxes[:, 3])

new_image = new_image[top-5:bottom+5, left-5:right+5]

boxes = boxes.tolist()
print(boxes)
find_contours()
boxes = np.asarray(boxes)

characters = []
for box in boxes:
    characters.append(new_image[box[1]:box[3], box[0]:box[2]])

text = pytesseract.image_to_string(image_inv)
print(text)
for image in characters:
    #cv2.resize(image, (round(image.shape[1] * 3), round(image.shape[0] * 3)))
    character = pytesseract.image_to_string(image)
    print("Character: ", character)
    cv2.imshow("Character", image)
    cv2.waitKey()
cv2.imshow("Image", new_image)
cv2.waitKey()