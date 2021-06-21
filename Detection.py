import numpy as np
from numpy import expand_dims
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import cv2
from mrcnn.model import mold_image
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from Utility.ImageProcessing import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import io
import time
import re
from difflib import SequenceMatcher
# Change path to your own tesseract installation.
# Tesseract installation: https://github.com/tesseract-ocr/tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# define the prediction configuration
class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "plate_cfg"
    # number of classes (background + kangaroo)
    NUM_CLASSES = 1 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    GENERATE_MASKS = False


def LoadModel():
    """
    Function to load the mask rcnn model. 

    :return: model
    :rtype: MaskRCNN 
    """
    cfg = PredictionConfig()
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
    model.load_weights('mask_rcnn_plate_cfg_vers2_00100.h5', by_name=True)
    return model, cfg


def predict_once(img, model, cfg, debug=False):
    """
    :param img: path to the image we want to detect plates on
    :param model: license plate detection model
    :param cfg: configuration for the model
    :type img: string
    :type model: .h5 file
    :type cfg: PredictionConfig
    :return boxes: detected bounding boxes
    :return image: the img
    :rtype boxes: list
    :rtype image: image
    """
    # load image and mask
    image = cv2.imread(img)
    # convert pixel values (e.g. center)
    scaled_image = mold_image(image, cfg)
    # convert image into one sample
    sample = expand_dims(scaled_image, 0)
    # make prediction
    yhat = model.detect(sample, verbose=0)[0]

    if debug:
        # show the figure
        # plot raw pixel data
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Predicted')
        print("Confidence score:", yhat['scores'])

        # define subplot
        ax = plt.gca()
        # plot each box
        for box in yhat['rois']:
            # get coordinates
            y1, x1, y2, x2 = box
            # calculate width and height of the box
            width, height = x2 - x1, y2 - y1
            # create the shape
            rect = Rectangle((x1, y1), width, height, fill=False, color='red')
            # draw the box
            ax.add_patch(rect)
        plt.show()
    return yhat['rois'], image


def Tesseract_OCR(img, lang):
    """
    Function to read text on images with pytesseract 

    :param img: the image that needs to be read from 
    :param lang: language for Tesseract to use 
    :type img: cv2 
    :type lang: string 

    :return plates: the plates that were detected
    :rtype plates: string list 
    """
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    plates = []
    for i in range(20, 70, 5):
        im_bw = cv2.threshold(im_gray, thresh + i, 255, cv2.THRESH_BINARY)

        plate_crop = Image.fromarray(im_bw[1])
        new_size = tuple(14*x for x in plate_crop.size)
        resized_img = plate_crop.resize(new_size, Image.ANTIALIAS)

        text = pytesseract.image_to_string(resized_img, lang=lang, config='--psm 10')
        text = re.sub('[\n-\x0c]',  '', text)
        text = re.sub('[^A-Za-z0-9 ]+', '', text)
        text = text.replace("  ", "-")
        text = text.replace(" ", "-")
        length_tester = text.replace("-", "")
        if len(length_tester) >= 6:
            if text[0] == "-":
                text = text[:0] + "" + text[0 +1:]               
            plates.append(re.sub('[\n-\x0c]',  '', text))
    plates = list(set(plates))
    return plates

def detect_and_return_plate(img, model, cfg):
    boxes, images = predict_once(img, model, cfg)
    first_c = first_crops(images, boxes)

    plates_list = []

    for crop in first_c:
        second_crop = secondCrop(crop)
        contrasted_img = increase_contrast_2(increase_contrast_1(second_crop))
        contrasted_img = apply_brightness_contrast(contrasted_img, brightness=-50, contrast=30)

        plates = Tesseract_OCR(contrasted_img, lang="plate_model_4")
        plates_list.append(plates)
    return plates_list
