from PIL import ImageFont, ImageDraw, Image
import cv2
import numpy as np
characters = [x for x in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ#$()+,-.=[]^{}']


# TODO: Add comments to all methods to describe code.


def create_base(width, length):
    base = np.full((length, width), 255, dtype="uint8")
    return base


def add_text(base, char, test):
    current_char = char + "  "
    img = Image.fromarray(base)
    draw = ImageDraw.Draw(img)
    for i in range(16):
        font = ImageFont.truetype("TFF/Roboto-MediumItalic.ttf", i+30)
        draw.text((10, i * 40 + i * 10 + 5), current_char * 32, font=font, fill=0)
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
        add_text(create_base(2000, 850), char, test)


if __name__ == "__main__":
    main(False)
