from mrcnn.utils import Dataset
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray


# class that defines and loads license plate dataset
class PlateDataset(Dataset):
    """
    Plate class for loading in the train/test sets
    Adding this line of text here as a test.
    """
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True, train_max=81):
        """
        :param dataset_dir: path to the directory containing the data
        :param is_train: whether we're loading the train set or test set
        :param train_max: the size of the train set
        :type dataset_dir: string
        :type is_train: bool
        :type train_max: int
        """
        # define one class
        self.add_class("dataset", 1, "plate")
        # define data locations
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'
        # find all images
        for filename in listdir(images_dir):
            # extract image id
            image_id = filename[:-4]
            # skip bad images
            # skip all images after 150 if we are building the train set
            if is_train and int(image_id) >= train_max:
                continue
            # skip all images before 150 if we are building the test/val set
            if not is_train and int(image_id) < train_max:
                continue
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
            # add to dataset
            self.add_image('dataset', image_id=image_id,
                           path=img_path, annotation=ann_path)

    # function to extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
        """
        :param filename: name of annotation file
        :type filename: str
        :return boxes: all the bounding boxes in an image
        :return width: width of a box
        :return height: height of a box
        :type boxes: list
        :type width: int
        :type height: int
        """
        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    # load the masks for an image
    def load_mask(self, image_id):
        """
        :param image_id: id of an image
        :type image_id: int
        :return class_ids: id of each class in the image
        :return masks: each mask found in the image
        :type class_ids: array
        :type masks: array
        """
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load XML
        boxes, w, h = self.extract_boxes(path)
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('plate'))
        return masks, asarray(class_ids, dtype='int32')

    # load an image reference
    def image_reference(self, image_id):
        """
        :param image_id: id of an image
        :type image_id: int
        :return path: path to the image
        :type path: string
        """
        info = self.image_info[image_id]
        return info['path']

# source: https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/
