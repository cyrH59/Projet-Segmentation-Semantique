#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 16:44:39 2022

@author: cyrilhannier
"""

import torch
import torchvision
import cv2
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image
from src.utils import draw_segmentation_map, get_outputs

# initialize the model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=91)
# set the computation device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load the model on to the computation device and set to eval mode
model.to(device).eval()

nbimage = 130; #nombre d'image correspond au frame.
treshold = 0.7;

s=0;
image_path = "images/framesv1/%d.png" % s
image = Image.open(image_path).convert("RGB")


for k in range(5):
    s=k*5
    image_path = "images/framesv1/%d.png" % s
    image = Image.open(image_path).convert("RGB")
    orig_image = image.copy()
    masks, boxes, labels, scores, scoresreturn = get_outputs(image, model, 0.7)
    result = draw_segmentation_map(orig_image, masks, boxes, labels)
    
    plt.cla()
    plt.imshow(result)
    plt.pause(0.01)
    # fig = plt.figure()
    # ax  = plt.subplot(111)
    # im  = ax.imshow(result)

    #   # Change image contents
    # newImData = np.array([[2,2],[2,2]])
    # im.set_data( newImData )
    # im.draw()
plt.show()
