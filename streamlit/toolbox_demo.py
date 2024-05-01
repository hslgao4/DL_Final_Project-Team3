import numpy as np
import pandas as pd
import random
from glob import glob
import seaborn as sns
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import shutil
from tqdm import tqdm
tqdm.pandas()
from PIL import Image
import io
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import joblib
from collections import defaultdict
import gc
from sklearn.model_selection import train_test_split
import pdb

import cv2
import torchvision
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import monai
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
import segmentation_models_pytorch as smp
import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2
import rasterio
from joblib import Parallel, delayed

'''Run-length-encode-and-decode '''
def rle_decode(mask_rle, shape):
    # s = mask_rle.split()
    # starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    # starts -= 1

    s = np.asarray(mask_rle.split(), dtype=int)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction, Returns numpy array, 1 - mask, 0 - background

def rle_encode(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)   # Returns run length as string formatted


'''U-Net module'''
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_h = x2.size()[2] - x1.size()[2]
        diff_w = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_h // 2, diff_h - diff_h // 2,
                        diff_w // 2, diff_w - diff_w // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels

        self.D_conv = DoubleConv(in_channels, channels)

        self.Down1 = Down(channels, 2*channels)
        self.Down2 = Down(2*channels, 4*channels)
        self.Down3 = Down(4*channels, 8*channels)
        self.Down4 = Down(8*channels, 8*channels)
        # self.Down5 = Down(16*48, 16*48)

        self.Up1 = Up(16*channels, 4*channels)
        self.Up2 = Up(8*channels, 2*channels)
        self.Up3 = Up(4*channels, channels)
        self.Up4 = Up(2*channels, channels)
        # self.Up5 = Up(2*48, 1*48)

        self.Out_Conv = OutConv(channels, out_channels)

    def forward(self, x):
        d0 = self.D_conv(x)
        d1 = self.Down1(d0)
        d2 = self.Down2(d1)
        d3 = self.Down3(d2)
        d4 = self.Down4(d3)
        # d5 = self.Down5(d4)

        mask1 = self.Up1(d4, d3)
        mask2 = self.Up2(mask1, d2)
        mask3 = self.Up3(mask2, d1)
        mask4 = self.Up4(mask3, d0)
        # mask5 = self.Up5(mask4, d0)

        logits = self.Out_Conv(mask4)

        return logits

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def UNET():
    model = UNet(in_channels=1, channels=24, out_channels=3)
    model.to(device)
    return model

def show_img(ax, img, mask=None):
    ax.imshow(img, cmap='bone')
    if mask is not None:
        ax.imshow(mask, alpha=0.5)
        handles = [Rectangle((0, 0), 1, 1, color=_c) for _c in
                   [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        ax.legend(handles, labels)
    ax.axis('off')

def plot_image_mask(image, mask):
    fig, ax = plt.subplots(figsize=(25, 5))
    images = image[0,].permute((1, 2, 0)).numpy()
    masks = mask[0,].permute((1, 2, 0)).numpy()

    show_img(ax, images, masks)
    plt.tight_layout()

    return fig

def demo_image(image):
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, -1)
    image = image.astype(np.float32) / 255
    image = np.transpose(image, (2, 0, 1))
    tensor_image = torch.tensor(image)
    return tensor_image.unsqueeze(0)


class FCN8s(nn.Module):

    def __init__(self, n_class=3):
        super(FCN8s, self).__init__()
        self.features_123 = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2

            # conv2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4

            # conv3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
        )
        self.features_4 = nn.Sequential(
            # conv4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
        )
        self.features_5 = nn.Sequential(
            # conv5 features
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/32
        )
        self.classifier = nn.Sequential(
            # fc6
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            # fc7
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            # score_fr
            nn.Conv2d(4096, n_class, 1),
        )
        self.score_feat3 = nn.Conv2d(256, n_class, 1)
        self.score_feat4 = nn.Conv2d(512, n_class, 1)
        self.upscore = nn.ConvTranspose2d(n_class, n_class, 16, stride=8,
                                              bias=False)
        self.upscore_4 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2,
                                              bias=False)
        self.upscore_5 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2,
                                              bias=False)

    def forward(self, x):
        feat3 = self.features_123(x)  #1/8
        feat4 = self.features_4(feat3)  #1/16
        feat5 = self.features_5(feat4)  #1/32

        score5 = self.classifier(feat5)
        upscore5 = self.upscore_5(score5)
        score4 = self.score_feat4(feat4)
        score4 = score4[:, :, 5:5+upscore5.size()[2], 5:5+upscore5.size()[3]].contiguous()
        score4 += upscore5

        score3 = self.score_feat3(feat3)
        upscore4 = self.upscore_4(score4)
        score3 = score3[:, :, 9:9+upscore4.size()[2], 9:9+upscore4.size()[3]].contiguous()
        score3 += upscore4
        h = self.upscore(score3)
        h = h[:, :, 28:28+x.size()[2], 28:28+x.size()[3]].contiguous()

        return h
def model_definition():
    model = FCN8s()
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=0, verbose=True)

    return model, optimizer, criterion, scheduler

def load_model(path):
    model, optimizer, criterion, scheduler = model_definition()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def show_img2(img, mask=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    plt.imshow(img, cmap='bone')

    if mask is not None:
        plt.imshow(mask, alpha=0.5)
        handles = [Rectangle((0, 0), 1, 1, color=_c) for _c in
                   [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        plt.legend(handles, labels)
    plt.axis('off')
    return plt.gcf()


def preprocess_image(image, device):
    image = torchvision.transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    return image

def predict(img, model):
    size = (224, 224)
    preds = []
    img = preprocess_image(img, device)
    with torch.no_grad():
        pred = model(img)
        pred = nn.Sigmoid()(pred)
    preds.append(pred)
    return img, preds