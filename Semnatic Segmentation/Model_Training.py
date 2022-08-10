import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random
import cv2

import CityScapes_labels as labels
from CityScapes import CityScapesInterface

import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.segmentation import *



# check GPU
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
print(device)


data_root = '/mnt/data/course/psarin/inm705' # path to data
list_of_classes = ['__bgr__', 'car', 'person'] # list of classes we wont to classify


train_interface_params = {
    'data_root': data_root,
    'list_of_classes': list_of_classes,
    'labels': labels,
    'phase': 'train',
    'task': 'semantic'
}

val_interface_params = {
    'data_root': data_root,
    'list_of_classes': list_of_classes,
    'labels': labels,
    'phase': 'val',
    'task': 'semantic'
}

test_interface_params = {
    'data_root': data_root,
    'list_of_classes': list_of_classes,
    'labels': labels,
    'phase': 'test',
    'task': 'semantic'
}

train_interface = CityScapesInterface(**train_interface_params) # train data
val_interface = CityScapesInterface(**val_interface_params) # validation data
test_interface = CityScapesInterface(**test_interface_params) # test data


train_dataloader = data.DataLoader(train_interface, batch_size=1, shuffle=True)
val_dataloader = data.DataLoader(val_interface, batch_size=1, shuffle=False)
test_dataloader = data.DataLoader(test_interface, batch_size=1, shuffle=False)


fcn_model = segmentation.fcn_resnet50(pretrained_backbone=False, pretrained=False, num_classes=3, aux_loss=True)
fcn_model=fcn_model.to(device)

# Load model with pre-trained backbone
fcn_model.load_state_dict(torch.load('./fcn_model_pretrained_backbone.pth'))
fcn_model.eval()

def predict(model, img_path):
    model.eval()
    im = Image.open(img_path)
    t_ = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.457, 0.407],
                    std=[0.229,0.224,0.225])])                    
    img = t_(im)    
    img.unsqueeze_(0)
    if device == torch.device("cuda"):
        img = img.to(device)
    # get the output from the model
    fcn_model.eval()
    bgr_img = np.array(Image.open(img_path))
    output = model(img)['out']
    out = output.argmax(1).squeeze_(0).detach().clone().cpu().numpy()
    color_array = np.zeros([out.shape[0], out.shape[1],3], dtype=np.uint8)
    print("Detected:")
    for id in np.unique(out):
        if id == 1:
            color_array[out==id] = [255,0,0]
        elif id == 2:
            color_array[out==id] = [0,255,0] 
    added_image = cv2.addWeighted(bgr_img, 0.5, color_array,0.6, 0)
    plt.imshow(added_image)
    plt.show()
    
    '''
    plt.imshow(im)
    plt.imshow(out, alpha=0.6)
    plt.title('Picture with Mask Appplied')
    plt.axis('off')
    plt.show()
    print(np.unique(out)[1:])
    '''
    
    
# move model to cuda
if device == torch.device('cuda'):
    fcn_model = fcn_model.to(device)

fcn_model.train() # train mode    
# optimizer
opt_pars = {'lr':1e-5, 'weight_decay':1e-3}
optimizer = torch.optim.Adam(list(fcn_model.parameters()),**opt_pars)
total_epochs = 1


checkpoint = torch.load("./checkpoints/model.pt")
fcn_model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']


m = nn.LogSoftmax(dim=1)
loss_function = nn.NLLLoss()

for e in range(total_epochs):
    epoch_loss = 0
    start_time = time.time()
    fcn_model.train()
    for id, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        idx, X, y = batch
        if device == torch.device('cuda'):
            X, y['mask'] = X.to(device), y['mask'].to(device)
            # list of images
            #images = [im for im in X]
            targets = []
            lab={}
            lab['mask'] = y['mask'].squeeze_(0)        
            targets.append(lab)
            # avoid empty objects
        if len(targets)>0:
            output = fcn_model(X)['out']
            loss = loss_function(m(output), targets[0]['mask'].unsqueeze(0).long())
            #print('Loss: ', loss.item())         
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
    
    epoch_loss = epoch_loss/len(train_dataloader)
    epoch_time = (time.time() - start_time) /60**1
    
    torch.save({
                'epoch': e,
                'model_state_dict': fcn_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                }, "./checkpoints/semantic_segmentation_v1.pt")
    
    print("Loss = {0:.4f} in epoch {1:d}. Time: {2:f}".format(epoch_loss, e, epoch_time))