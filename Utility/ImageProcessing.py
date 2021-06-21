import cv2
import numpy as np
import matplotlib.pyplot as plt

#------------------------------CROPPING---------------------------------#
def first_crops(img, boxes):
    """
    :param img: image we're cropping
    :param boxes: bounding boxes possibly containing plates
    :type img: image
    :type boxes: list
    :return crops: return made crops
    :rtype crops: list
    """
    crops = []
    for box in boxes:
        # get coordinates
        y1, x1, y2, x2 = box
        firstCrop = img[y1:y2, x1:x2]
        #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        crops.append(firstCrop)
    return crops


def secondCrop(img):
    """
    :param img: cropped image
    :type img: image
    :return secondCrop:
    :rtype secondCrop: image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    if(len(areas) != 0):
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        x, y, w, h = cv2.boundingRect(cnt)
        #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        secondCrop = img[y:y + h, x:x + w]
    else:
        secondCrop = img
    return secondCrop

#----------------------SHARPEN FUNCTIONS---------------------------------------#
def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

#----------------------CONTRAST FUNCTONS---------------------------------------#
def increase_contrast_1(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    plt.imshow(final)

    return final

def increase_contrast_2(img):
    alpha = 1.5
    beta = 20
    output = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)
    plt.imshow(output)
    return output

def increase_contrast_3(img):
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(cv2.cvtColor(grayImage, cv2.COLOR_BGR2RGB))
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 210, 255, cv2.THRESH_BINARY)
    #plt.imshow(blackAndWhiteImage)
    return blackAndWhiteImage


def apply_brightness_contrast(input_img, brightness=64, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf