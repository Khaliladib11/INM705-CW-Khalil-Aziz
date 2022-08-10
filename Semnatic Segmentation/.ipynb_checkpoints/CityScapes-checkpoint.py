import os

import numpy as np
import matplotlib.pyplot as plt
import json

import torch
from torch.utils import data
from torchvision import transforms

from PIL import Image


class CityScapesInterface(data.Dataset):

    # constructor
    def __init__(self, 
                 data_root,
                 list_of_classes,
                 labels,
                 phase,
                 task,
                 mean_transform=[0.407, 0.457, 0.485],
                 std_transform=[0.229,0.224,0.225]):
        
        self.data_root = data_root
        self.list_of_classes = list_of_classes
        self.label2id = labels.label2id
        self.id2label = labels.id2label

        if phase == 'test':
            self.phase = 'test'
        elif phase == 'val':
            self.phase = 'val'
        else:
            self.phase = 'train'

        if task == 'instance':
            self.task = 'instance'
        elif task == 'semantic':
            self.task = 'semantic'
        else:
            self.task = 'Object Detection'
           
        self.mean_transform = mean_transform
        self.std_transform = std_transform

        self.cities = os.listdir(os.path.join(self.data_root, 'leftImg8bit', self.phase))
        self.imgs = self.__getAllImages()
        self.semantic_masks = self.__load_semantic_masks()
        self.instance_masks = self.__load_instance_masks()
        self.annotation_files = self.__load_annotation_files()
        self.__create_classes_helper()

    # method to create dict for classes
    def __create_classes_helper(self):
        self.class_to_idx = {}
        self.idx_to_class = {}
        for idz, c in enumerate(self.list_of_classes[1:]):
            assert c in self.label2id.keys()
            self.class_to_idx[self.label2id[c]] = idz + 1
            self.idx_to_class[idz + 1] = self.label2id[c]

    # Load all the images
    def __getAllImages(self):
        imgs = []
        for city in self.cities:
            city_imgs = os.listdir(os.path.join(self.data_root, 'leftImg8bit', self.phase, city))
            imgs.extend(city_imgs)

        return imgs
    

    # Load semantic masks
    def __load_semantic_masks(self):
        semantic_masks = []
        for city in self.cities:
            masks = os.listdir(os.path.join(self.data_root, 'gtFine_trainvaltest/gtFine', self.phase, city))
            for file in masks:
                if file.endswith('labelIds.png'):
                    semantic_masks.append(file)

        return semantic_masks

    # Load instance masks
    def __load_instance_masks(self):
        instance_masks = []
        for city in self.cities:
            masks = os.listdir(os.path.join(self.data_root, 'gtFine_trainvaltest/gtFine', self.phase, city))
            for file in masks:
                if file.endswith('instanceIds.png'):
                    instance_masks.append(file)

        return instance_masks

    # Load annotation files
    def __load_annotation_files(self):
        annotation_files = []
        for city in self.cities:
            files = os.listdir(os.path.join(self.data_root, 'gtFine_trainvaltest/gtFine', self.phase, city))
            for file in files:
                if file.endswith('.json'):
                    annotation_files.append(file)

        return annotation_files

    def image_transform(self, img):
        t_ = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = self.mean_transform,
                                 std = self.std_transform)
        ])
        img = t_(img)
        return img

    # Load an image
    def load_img(self, idx, transform=True):
        img = self.imgs[idx]
        city_name = img.split('_')[0]
        im = Image.open(os.path.join(self.data_root, 'leftImg8bit', self.phase, city_name, img))
        if transform:
            im = self.image_transform(im)    
        return im

    # Get Bounding boxes
    def load_bounding_boxes(self, idx):
        img_file_name = self.imgs[idx]
        img_file_name = img_file_name.split("_")
        annotation_file = img_file_name[0]+'_'+img_file_name[1]+'_'+img_file_name[2]+'_gtFine_polygons.json'
        #annotation_file = self.annotation_files[idx]
        city_name = annotation_file.split('_')[0]
        annotation_path = os.path.join(self.data_root, 'gtFine_trainvaltest/gtFine', self.phase, city_name, annotation_file)
        bbox_json = open(annotation_path)
        data = json.load(bbox_json)
        objects = data['objects']
        list_of_objects = []
        bboxes = []
        for obj in objects:
            # print(obj)
            if obj['label'] in self.list_of_classes[1:]:
                # print(obj)
                list_of_objects.append(self.class_to_idx[self.label2id[obj['label']]])
                x, y = zip(*obj['polygon'])
                min_x, max_x = min(x), max(x)
                min_y, max_y = min(y), max(y)
                bbox = [min_x, min_y, max_x, max_y]
                bboxes.append(bbox)
        label = {}
        classes = torch.tensor(list_of_objects, dtype=torch.int64)
        label['labels'] = classes
        label['boxes'] = torch.tensor(bboxes, dtype=torch.float32)
        return label

    # Get Semantic Mask
    def load_semantic_mask(self, idx):
        img_file_name = self.imgs[idx]
        img_file_name = img_file_name.split("_")
        semantic_mask_file = img_file_name[0]+'_'+img_file_name[1]+'_'+img_file_name[2]+'_gtFine_labelIds.png'
        #semantic_mask_file = self.semantic_masks[idx]
        city_name = semantic_mask_file.split('_')[0]
        mask = np.array(Image.open(os.path.join(self.data_root, 'gtFine_trainvaltest/gtFine', self.phase, city_name, semantic_mask_file)))
        unique_values = np.unique(mask)
        for label in unique_values:
            _l = self.id2label[label]
            if _l not in self.list_of_classes:
                mask[mask == label] = 0

        for value in np.unique(mask)[1:]:
            mask[mask == value] = self.class_to_idx[value]

        mask = torch.tensor(mask, dtype=torch.uint8)
        return mask

    # method to get instance segmantation mask
    def load_instance_masks(self, idx):
        
        img_file_name = self.imgs[idx]
        img_file_name = img_file_name.split("_")
        instance_mask_file = img_file_name[0]+'_'+img_file_name[1]+'_'+img_file_name[2]+'_gtFine_instanceIds.png'
        
        #instance_mask_file = self.instance_masks[idx]
        city_name = instance_mask_file.split('_')[0]
        mask = np.array(Image.open(os.path.join(self.data_root, 'gtFine_trainvaltest/gtFine', self.phase, city_name, instance_mask_file)), dtype=np.uint8)
        mask = torch.tensor(mask, dtype=torch.uint8)

        _mask_segmentation = self.load_semantic_mask(idx)
        # plt.imshow(_mask_segmentation)

        dict_values = dict()
        _instance_mask = mask * _mask_segmentation
        for ins in np.unique(_mask_segmentation)[1:]:
            y = _mask_segmentation.clone()
            y[y != ins] = 0
            _mask = mask * y
            for pixel in np.unique(_mask)[1:]:
                dict_values[pixel] = ins
                # print(len(segmented_masks_list))
                # print(ins, ' ', list_of_classes[ins])

        _instances = np.unique(_instance_mask)
        list_of_masks = []
        for _m in _instances[1:]:
            _mask = torch.zeros(mask.shape, dtype=torch.uint8)
            # print(_m, ' ',list_of_classes[dict_values[_m]])
            _mask[_instance_mask == _m] = dict_values[_m]
            # plt.imshow(_mask)
            # plt.show()
            list_of_masks.append(_mask)

        return list_of_masks

    # method to return bounding boxes and semantic mask
    def semantic_segmentation_task(self, idx):
        label = self.load_bounding_boxes(idx)
        semantic_mask = self.load_semantic_mask(idx)
        
        label['mask'] = semantic_mask
        return label

    # method to return bounding boxes and instance masks
    def instance_segmentation_task(self, idx):
        img_file_name = self.imgs[idx]
        img_file_name = img_file_name.split("_")
        instance_mask_file = img_file_name[0]+'_'+img_file_name[1]+'_'+img_file_name[2]+'_gtFine_instanceIds.png'
        
        #instance_mask_file = self.instance_masks[idx]
        city_name = instance_mask_file.split('_')[0]
        instance_mask = np.array(Image.open(os.path.join(self.data_root, 'gtFine_trainvaltest/gtFine', self.phase, city_name, instance_mask_file)), dtype=np.uint8)
        semantic_mask_img = self.load_semantic_mask(idx)

        final_mask = semantic_mask_img * instance_mask

        label = self.load_bounding_boxes(idx)

        bboxes = label['boxes']
        masks = []

        for idx in range(len(bboxes)):
            bbox = bboxes[idx]
            bbox = [int(x) for x in bbox]
            aux_array = np.zeros([final_mask.shape[0], final_mask.shape[1]])
            crop = final_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            aux_array[bbox[1]:bbox[3], bbox[0]:bbox[2]] = crop
            # aux_array[aux_array != unique_values[idx_unique_value]] = 0
            # mask = aux_array[aux_array == unique_values[idx_unique_value]]
            aux_array = torch.tensor(aux_array, dtype=torch.uint8)
            masks.append(aux_array)

        label['masks'] = masks
        return label

    # get length
    def __len__(self):
        return len(self.imgs)

    # get an item
    def __getitem__(self, idx):
        X = self.load_img(idx)

        if self.task == 'instance':
            y = self.instance_segmentation_task(idx)

        elif self.task == 'semantic':
            y = self.semantic_segmentation_task(idx)

        elif self.task == 'Object Detection':
            y = self.load_bounding_boxes(idx)

        return idx, X, y
