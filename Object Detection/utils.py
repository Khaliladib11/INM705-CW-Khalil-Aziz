import os
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torchvision
import torchvision.ops.boxes as bops
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchmetrics.detection.mean_ap import MeanAveragePrecision


# function to load ms coco classes from txt file in a suitable format
def load_coco_classes(file_path):
    coco_object_categories = []
    coco_classes = {}
    with open(file_path, 'r') as f:
        for id, category in enumerate(f.readlines()):
            category = re.sub('[\d]+\W+', '', category.rstrip())
            coco_object_categories.append(category)
            coco_classes[category] = id

    return coco_classes, coco_object_categories


# function to load Faster-RCNN model from torchvision repository takes as input the resnetbackbone we want to use,
# the number of classes we want to classify (number of classes plus background) pretrained ad pretrained backbone two
# boolean params, and model path if we want to load the weight manually.
def load_faster_rcnn(backbone="resnet50", num_classes=3, pretrained=True, pretrained_backbone=True, model_path=None):
    assert backbone in ['resnet50', 'resnet101',
                        'resnet152'], "Choose backbone from ['resnet50', 'resnet101', 'resnet152']"
    model_args = {
        'pretrained': pretrained,
        'pretrained_backbone': pretrained_backbone,
        'box_score_thresh': 0.5,
        'num_classes': 91,  # Shouldn't this be equal to num_classes(the argument to the function)?
        'rpn_batch_size_per_image': 256,
        'box_batch_size_per_image': 256}
    if backbone == 'resnet50':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(**model_args)
        if model_path is not None and pretrained is False:
            model.load_state_dict(torch.load(model_path))
        model.eval()

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    else:
        resnet_backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
            backbone,
            pretrained=pretrained_backbone)
        model = FasterRCNN(resnet_backbone, num_classes=num_classes)
        model.eval()

    return model


def export_model(model, path):
    torch.save(model.state_dict(), path)


def save_model(model, epoch, training_loss, validation_loss, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'training_loss': training_loss,
        # 'validation_loss': validation_loss,
    }, os.path.join(checkpoint_path, 'model.pth'))


def load_model(model, checkpoint_path):
    assert os.path.exists(checkpoint_path), "File not found"

    checkpoint = torch.load(os.path.join(checkpoint_path, 'model.pth'))
    model.load_state_dict(checkpoint['model'])
    epoch = checkpoint['epoch']
    training_loss = checkpoint['training_loss']
    # validation_loss = checkpoint['validation_loss']

    return epoch, model, training_loss


def train_model(model, train_loader, optimizer, epochs, device, checkpoint_path, load_checkpoint=None,
                show_every=1000):
    model.to(device)
    if load_checkpoint:
        # e, model, training_loss, validation_loss = load_model(model, load_checkpoint)
        e, model, training_loss = load_model(model, load_checkpoint)
    else:
        training_loss = []
        # validation_loss = []
        e = 0

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    file = open(os.path.join(checkpoint_path, "outputs.txt"), "a")

    start_time = time.time()
    for epoch in range(e, epochs):

        train_epoch_loss = 0
        val_epoch_loss = 0

        model.train()

        for i, batch in enumerate(train_loader):
            idx, X, y = batch
            X, y['labels'], y['boxes'] = X.to(device), y['labels'].to(device), y['boxes'].to(device)

            model.zero_grad()
            images = [im for im in X]

            targets = []
            lab = {}
            lab['boxes'] = y['boxes'].squeeze_(0)
            lab['labels'] = y['labels'].squeeze_(0)
            targets.append(lab)

            if len(targets) > 0:
                t_loss = model(images, targets)
                total_loss = 0
                # print(loss)
                # for k in loss.keys():
                #    total_loss += loss[k]
                total_loss = sum(loss for loss in t_loss.values())
                total_loss.backward()
                optimizer.step()
                # print(total_loss.item())
                train_epoch_loss += total_loss.item()

            if i % show_every == 0:
                state = "Epoch: {:15} || Step: {:15} || Average Training Loss: {:.4f}".format(
                    '[{:d}/{:d}]'.format(epoch, epochs),
                    '[{:d}/{:d}]'.format(i, len(train_loader)),
                    train_epoch_loss / (i + 1))
                print(state)
                file.write(state + '\n')
                file.flush()

        train_epoch_loss /= len(train_loader)

        training_loss.append(train_epoch_loss)

        """
        # model.eval()

        for i, batch in enumerate(val_loader):
            idx, X, y = batch
            X, y['labels'], y['boxes'] = X.to(device), y['labels'].to(device), y['boxes'].to(device)

            model.zero_grad()
            images = [im for im in X]

            targets = []
            lab = {}
            lab['boxes'] = y['boxes'].squeeze_(0)
            lab['labels'] = y['labels'].squeeze_(0)
            targets.append(lab)

            if len(targets) > 0:
                loss = model(images, targets)
                val_loss = 0
                # print(loss)
                for k in loss.keys():
                    val_loss += loss[k]
                with torch.no_grad():
                    v_loss = model(images, targets)

                val_loss = sum(loss for loss in v_loss.values())
                val_epoch_loss += val_loss.item()

            if i % show_every == 0:
                state = "Epoch: {:15} || Step: {:15} || Average Validation Loss: {:.4f}".format(
                    '[{:d}/{:d}]'.format(epoch, epochs),
                    '[{:d}/{:d}]'.format(i, len(val_loader)),
                    val_epoch_loss / (i + 1))
                print(state)
                file.write(state + '\n')
                file.flush()

        val_epoch_loss /= len(val_loader)

        validation_loss.append(val_epoch_loss)
        """

        epoch_time = (time.time() - start_time) / 60 ** 1

        # state = "Epoch: [{0:d}/{1:d}] || Training Loss = {2:.2f} || Validation Loss: {3:.2f} || Time: {4:f}" \
        #    .format(epoch, epochs, train_epoch_loss, val_epoch_loss, epoch_time)

        state = "Epoch: [{0:d}/{1:d}] || Training Loss = {2:.2f} || Time: {4:f}" \
            .format(epoch, epochs, train_epoch_loss, epoch_time)

        print(100 * "*")
        print(state)
        print(100 * "*")
        file.write(100 * "*" + '\n')
        file.write(state + '\n')
        file.write(100 * "*" + '\n')
        file.flush()

        save_model(model, epoch, training_loss, checkpoint_path)

    file.close()
    return training_loss


def predict(model, img, target_classes, device):
    model.to(device)
    model.eval()
    # img = Image.open(img)
    img = np.array(img)
    img_tensor = transforms.ToTensor()(img).to(device)
    out = model([img_tensor])
    scores = out[0]['scores'].cpu().detach().numpy()
    bboxes = out[0]['boxes'].cpu().detach().numpy()
    classes = out[0]['labels'].cpu().detach().numpy()
    fig, ax = plt.subplots()
    ax.imshow(img)
    color_map = ['b', 'r', 'y', 'g']
    for i in range(len(classes)):
        if scores[i] > 0.75:
            bbox = bboxes[i]
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                     edgecolor=color_map[classes[i]], facecolor="none")
            ax.add_patch(rect)
            ax.text((bbox[0] + bbox[2]) / 2 - 30, bbox[1] - 5, target_classes[classes[i]], c=color_map[classes[i]])

    plt.axis('off')
    plt.show()


# compare different models, takes as input training losses list and titles
def compare_models(training_models_loses, titles):
    assert len(training_models_loses) == len(titles), "Length of training list and titles list have to be equals."

    for i in range(len(training_models_loses)):
        plt.plot(training_models_loses[i], label=titles[i])

    plt.legend(loc='upper right')
    plt.title('Comparing Object Detection with different Resnet backbone')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


# function to plot the training loss vs validation loss
def plot_loss(training_loss, validation_loss):
    plt.plot(training_loss, label="Training Loss")
    plt.plot(validation_loss, label="Validation Loss")
    plt.legend(loc='upper right')
    plt.title('Training vs Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


################################################## Evaluation ##################################################

# method to calculate the intersection over union between two boxes, takes as input two lists, returns iou
def get_iou(box1, box2):
    assert box1[0] > box1[2], "x0 > x1 in Box1, How is that possible?"
    assert box1[1] > box1[3], "y0 > y1 in Box1, How is that possible?"
    assert box2[0] > box2[2], "x0 > x1 in Box2, How is that possible?"
    assert box2[1] > box2[3], "y0 > y1 in Box2, How is that possible?"

    box1 = torch.tensor([box1], dtype=torch.float)
    box2 = torch.tensor([box2], dtype=torch.float)
    iou = bops.box_iou(box1, box2)
    return iou


def evaluate_mAP(model, loader, device):
    model.to(device)
    model.eval()
    metric = MeanAveragePrecision()
    for i, batch in enumerate(loader):
        idx, X, y = batch
        X, y['labels'], y['boxes'] = X.to(device), y['labels'].to(device), y['boxes'].to(device)

        model.zero_grad()
        images = [im for im in X]

        targets = []
        lab = {}
        lab['boxes'] = y['boxes'].squeeze_(0)
        lab['labels'] = y['labels'].squeeze_(0)
        targets.append(lab)

        if len(targets) > 0:
            with torch.no_grad():
                pred = model(images, targets)
                if len(pred[0]['boxes']) > 0:
                    metric.update(pred, [lab])

    return metric.compute()
