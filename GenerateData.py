from PIL import ImageFont, ImageDraw, Image
import cv2
import numpy as np
characters = [x for x in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ#$()+,-.:=[]^{}']


def create_base(size):
    base = np.full((size, size), 255, dtype="uint8")
    return base


def add_text(base, char, test):
    current_char = char + " "
    img = Image.fromarray(base)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("TFF/Roboto-MediumItalic.ttf", 30)
    for i in range(30):
        draw.text((5, i*40), current_char * 32, font=font, fill=0)
    img = np.array(img)
    upper = False
    print(char)
    if char.isupper():
        upper = True
    extra = ""
    if test:
        extra = "_test"
    cv2.imwrite("PAGES/page_{0}{1}.png".format(char.lower() + "_upper" if upper else char, extra), img)
    return img


def display_image(img):
    cv2.imshow("Image", img)
    cv2.waitKey()


def main(test):
    for char in characters:
        add_text(create_base(1250), char, test)

if __name__ == "__main__":
    main(False)
