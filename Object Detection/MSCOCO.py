import os
import json
from collections import deque
import re
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random


class MSCOCO(data.Dataset):

    def __init__(self, root_path, list_of_classes, target_classes, stage='train', annotation_json_path=None):
        """
        MSCOCO constructor, inherited from data.Dataset
        :param root_path: The root for the original data.
        :param list_of_classes: dict contains the list of all classes with idx in the dataset (COCO has 91 classes).
        :param target_classes: list contains the names of the objects that we want to detect plus background.
        :param stage: to specify in which stage we are, to load the correct annotations.
        :param annotation_json_path: ready json annotations file, to save some time.
        """

        self.root_path = root_path

        self.list_of_classes = list_of_classes
        self.target_classes = target_classes

        self._annotation_helper()  # helper function

        assert stage in ['train', 'val', 'test'], "stage has to be in ['train', 'val', 'test']"
        # Training Stage
        if stage == 'train' or stage == 'val':
            self.data_path = os.path.join(self.root_path, 'train2017')
            self.annotation_path = os.path.join(self.root_path,
                                                'annotations_trainval2017/annotations/instances_train2017.json')

            # if we pass the annotations as params, just load them and save time
            if annotation_json_path:
                assert annotation_json_path.endswith(
                    'train_annotation.json'), f"You selected {stage} phase, upload Train annotations."
                self.annotations = json.load(open(annotation_json_path))
            # if no, load them using the get annotations method
            else:
                self.annotations = self.get_annotations()

        # Testing Stage
        elif stage == 'test':
            self.data_path = os.path.join(self.root_path, 'val2017')
            self.annotation_path = os.path.join(self.root_path,
                                                'annotations_trainval2017/annotations/instances_val2017.json')

            if annotation_json_path:
                assert annotation_json_path.endswith(
                    'val_annotation.json'), f"You selected {stage} phase, upload {stage} annotations."
                self.annotations = json.load(open(annotation_json_path))
            # if no, load them using the get annotations method
            else:
                self.annotations = self.get_annotations()

        # finally, after loading the annotations, load the images that contain the pictures we want
        all_imgs = self.get_imgs()

        random.seed(705)
        if stage == 'train':
            self.imgs = random.sample(all_imgs[0:45_000], 30_000)
        elif stage == 'val':
            self.imgs = random.sample(all_imgs[45_001:-1], 5_000)
        elif stage == 'test':
            self.imgs = all_imgs

    # helper method to create two dict for the target objects
    # index to class and class to index
    def _annotation_helper(self):
        # TODO : Check idx
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.coco_to_idx = {}
        for cls in self.target_classes:
            self.class_to_idx[cls] = self.list_of_classes[cls]
            self.idx_to_class[self.list_of_classes[cls]] = cls
            # self.class_to_idx[cls] = self.target_classes.index(cls)
            # self.idx_to_class[self.target_classes.index(cls)] = cls

    # extract images file names from annotations
    def get_imgs(self):
        imgs = []
        for anno in self.annotations:
            file_name = str(anno)
            file_name = file_name.zfill(12) + '.jpg'
            imgs.append(file_name)

        return imgs

    # method to load images, takes index as param
    def load_img(self, idx, transform=True):
        # load the image as Image object, convert it to RGB since we have grayscale images
        img = Image.open(os.path.join(self.data_path, self.imgs[idx])).convert('RGB')

        # if transform is true, apply it
        # we use this in training phase
        if transform:
            img = self.image_transform(img)
        return img

    # static method to apply image transforms on Image object
    @staticmethod
    def image_transform(img):
        t_ = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.407, 0.457, 0.485],
                                 std=[0.229, 0.224, 0.225])
        ])
        img = t_(img)
        return img

    # method to extract our target objects from the annotations file
    def get_annotations(self):
        data = json.load(open(self.annotation_path))
        # create the dic
        images_data = deque(data['images'])

        annotations = {}
        for img in images_data:
            annotations[str(img['id'])] = []

        annotations_data = deque(data['annotations'])

        for ann in annotations_data:
            if ann['category_id'] in self.idx_to_class:
                annotations[str(ann['image_id'])].append((ann['bbox'], ann['category_id']))

        for anno in annotations.copy():
            if len(annotations[anno]) == 0:
                annotations.pop(anno)

        return annotations

    # method to get the bounding boxes in each image, takes as input the index of the image
    def get_bboxes(self, idx):
        file_name = self.imgs[idx].split('.')[0].lstrip('0')
        label = {}
        classes = []
        bboxes = []
        for annotation in self.annotations[file_name]:
            box_coco = annotation[0]
            box = [box_coco[0], box_coco[1], box_coco[0] + box_coco[2], box_coco[1] + box_coco[3]]
            if box[2] - box[0] > 0 and box[3] - box[1] > 0:
                bboxes.append(box)
                classes.append(self.target_classes.index(self.idx_to_class[annotation[1]]))

        label['labels'] = torch.tensor(classes)
        label['boxes'] = torch.tensor(bboxes, dtype=torch.float32)
        return label

    # method to display an image with the corresponding bounding boxes and classes
    def display_img(self, idx):
        img = self.load_img(idx, False)  # load the image
        label = self.get_bboxes(idx)  # get the bounding boxes
        classes = [self.target_classes[idx] for idx in label['labels'].numpy()]  # get classes
        print(classes)
        bboxes = label['boxes']

        color_map = ['b', 'r', 'y', 'g']  # color map, to draw bounding box with different color for different object

        # plot the image
        fig, ax = plt.subplots()
        ax.imshow(img)
        for i in range(len(classes)):
            bbox = bboxes[i]
            # for bbox in bboxes:
            class_ = self.class_to_idx[classes[i]]  # to load to correspond color map
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                     edgecolor=color_map[class_],
                                     facecolor="none")
            ax.add_patch(rect)

        plt.axis('off')
        plt.show()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        X = self.load_img(idx)
        y = self.get_bboxes(idx)
        return idx, X, y
