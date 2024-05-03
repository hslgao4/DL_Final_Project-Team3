from colorama import Fore, Back, Style
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models

c_  = Fore.GREEN
sr_ = Style.RESET_ALL
import torch
from tqdm import tqdm
tqdm.pandas()
import copy
from collections import defaultdict
import gc
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# PyTorch
import torch.nn as nn
from torch.optim import lr_scheduler, optimizer, Optimizer
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

from glob import glob

import cv2
import pandas as pd
import numpy as np
import random
import torch.nn.functional as F
import os

from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

seed = 42
debug = False
model_name = 'Unet'
train_bs = 128
valid_bs = train_bs * 2
img_size = (224, 224)
n_epochs = 10
LR = 0.0001
scheduler = 'CosineAnnealingLR'
n_accumulate = max(1, 32 // train_bs)
n_fold = 5
fold_selected = 1
num_classes = 3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DIR = '/home/ubuntu/team3/Project_team3/data/'

def read_data():
    print(os.getcwd())
    df = pd.read_csv(DIR + 'train.csv')
    print(df.shape)

    # EDA
    df.rename(columns={'class': 'class_name'}, inplace=True)
    # --------------------------------------------------------------------------
    df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].replace("case", "")))
    df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].replace("day", "")))
    df["slice"] = df["id"].apply(lambda x: x.split("_")[3])
    # --------------------------------------------------------------------------
    TRAIN_DIR = DIR + "train"
    all_train_images = glob(os.path.join(TRAIN_DIR, "**", "*.png"), recursive=True)
    x = all_train_images[0].rsplit("/", 4)[0]  ## ../input/uw-madison-gi-tract-image-segmentation/train

    path_partial_list = []
    for i in range(0, df.shape[0]):
        path_partial_list.append(os.path.join(x,
                                              "case" + str(df["case"].values[i]),
                                              "case" + str(df["case"].values[i]) + "_" + "day" + str(
                                                  df["day"].values[i]),
                                              "scans",
                                              "slice_" + str(df["slice"].values[i])))
    df["path_partial"] = path_partial_list
    # --------------------------------------------------------------------------
    path_partial_list = []
    for i in range(0, len(all_train_images)):
        path_partial_list.append(str(all_train_images[i].rsplit("_", 4)[0]))

    tmp_df = pd.DataFrame()
    tmp_df['path_partial'] = path_partial_list
    tmp_df['path'] = all_train_images

    # --------------------------------------------------------------------------
    df = df.merge(tmp_df, on="path_partial").drop(columns=["path_partial"])
    # --------------------------------------------------------------------------
    df["width"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[1]))
    df["height"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[2]))
    # --------------------------------------------------------------------------
    del x, path_partial_list, tmp_df
    # --------------------------------------------------------------------------
    df.shape

    df_train = pd.DataFrame({'id': df['id'][::3]})

    df_train['large_bowel'] = df['segmentation'][::3].values
    df_train['small_bowel'] = df['segmentation'][1::3].values
    df_train['stomach'] = df['segmentation'][2::3].values

    df_train['path'] = df['path'][::3].values
    df_train['case'] = df['case'][::3].values
    df_train['day'] = df['day'][::3].values
    df_train['slice'] = df['slice'][::3].values
    df_train['width'] = df['width'][::3].values
    df_train['height'] = df['height'][::3].values

    df_train.reset_index(inplace=True, drop=True)
    df_train.fillna('', inplace=True);
    df_train['count'] = np.sum(df_train.iloc[:, 1:4] != '', axis=1).values
    df_train.sample(5)

    df_train.to_csv(DIR + 'train/final_df.csv')

    skf = StratifiedGroupKFold(n_splits=n_fold, shuffle=True, random_state=42)
    for fold, (_, val_idx) in enumerate(skf.split(X=df_train, y=df_train['count'], groups=df_train['case']), 1):
        df_train.loc[val_idx, 'fold'] = fold

    df_train['fold'] = df_train['fold'].astype(np.uint8)

    train_ids = df_train[df_train["fold"] != fold_selected].index
    valid_ids = df_train[df_train["fold"] == fold_selected].index

    df_train.groupby('fold').size()

#    train_dataset = BuildDataset(df_train[df_train.index.isin(train_ids)], transforms=data_transforms['train'])
#    valid_dataset = BuildDataset(df_train[df_train.index.isin(valid_ids)], transforms=data_transforms['valid'])

    train_dataset = BuildDataset(df_train[df_train.index.isin(train_ids)])
    valid_dataset = BuildDataset(df_train[df_train.index.isin(valid_ids)])


    train_loader = DataLoader(train_dataset, batch_size=train_bs, num_workers=4, shuffle=True, pin_memory=True,
                              drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=valid_bs, num_workers=4, shuffle=False, pin_memory=True)

    imgs, msks = next(iter(train_loader))
    imgs.size(), msks.size()

    return df_train, valid_ids

def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction

class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, df, subset="train", transforms=None):
        self.df = df
        self.subset = subset
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        masks = np.zeros((img_size[0], img_size[1], 3), dtype=np.float32)
        img_path = self.df['path'].iloc[index]
        w = self.df['width'].iloc[index]
        h = self.df['height'].iloc[index]
        img = self.__load_img(img_path)
        if self.subset == 'train':
            for k, j in zip([0, 1, 2], ["large_bowel", "small_bowel", "stomach"]):
                rles = self.df[j].iloc[index]
                mask = rle_decode(rles, shape=(h, w, 1))
                mask = cv2.resize(mask, img_size)
                masks[:, :, k] = mask

        masks = masks.transpose(2, 0, 1)
        img = img.transpose(2, 0, 1)

        if self.subset == 'train':
            return torch.tensor(img), torch.tensor(masks)
        else:
            return torch.tensor(img)

    def __load_gray_img(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        img = (img - img.min()) / (img.max() - img.min()) * 255.0
        img = cv2.resize(img, img_size)
        img = np.expand_dims(img, axis=-1)
        img = img.astype(np.float32) / 255.
        return img

    def __load_img(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = (img - img.min()) / (img.max() - img.min()) * 255.0
        img = cv2.resize(img, img_size)
        img = np.tile(img[..., None], [1, 1, 3])  # gray to rgb
        img = img.astype(np.float32) / 255.
        return img
class FCNVGG(nn.Module):
    def __init__(self):
        super(FCNVGG, self).__init__()

        vgg16 = models.vgg16(pretrained=True)
        self.vgg_features = vgg16.features

        # Freeze the VGG16 layers
        for param in self.vgg_features.parameters():
            param.requires_grad = False

        # Additional layers
        input_size = 7
        target_size = 224

        # Calculate the kernel_size and padding
        kernel_size = 2 * (target_size - input_size) + 1
        padding = kernel_size // 2

        # Define conv5 with adjusted parameters
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=3, kernel_size=kernel_size, padding=padding)
        nn.init.kaiming_normal_(self.conv5.weight)

        # Transpose convolution layers
        self.trans_conv1 = nn.ConvTranspose2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1)
        nn.init.kaiming_normal_(self.trans_conv1.weight)

        self.trans_conv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        nn.init.kaiming_normal_(self.trans_conv2.weight)

        self.trans_conv3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)
        nn.init.kaiming_normal_(self.trans_conv3.weight)

        self.trans_conv4 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1)
        nn.init.kaiming_normal_(self.trans_conv4.weight)

        self.conv6 = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv6.weight)

        self.conv7 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv7.weight)

    def forward(self, x):
        x = self.vgg_features(x)
        x = self.conv5(x)
        x = self.trans_conv1(x)
        x = self.trans_conv2(x)
        x = self.trans_conv3(x)
        x = self.trans_conv4(x)
        x = self.conv6(x)
        x = self.conv7(x)
        return x
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
    model = FCNVGG()
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=0, verbose=True)

    return model, optimizer, criterion, scheduler
def load_model(path):
    model, optimizer, criterion, scheduler = model_definition()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def show_img(img, mask=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    plt.imshow(img, cmap='bone')

    if mask is not None:
        plt.imshow(mask, alpha=0.5)
        handles = [Rectangle((0, 0), 1, 1, color=_c) for _c in
                   [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        plt.legend(handles, labels)
    plt.axis('off')
def plot_batch(imgs, msks, size=3):
    plt.figure(figsize=(5*5, 5))
    for idx in range(size):
        plt.subplot(1, 5, idx+1)
        img = imgs[idx,].permute((1, 2, 0)).numpy()
        msk = msks[idx,].permute((1, 2, 0)).cpu().numpy()
        show_img(img, msk)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    for i in range(1):
        df_train, valid_ids = read_data()
        test_dataset = BuildDataset(df_train[df_train.index.isin(valid_ids)],)
        test_loader = DataLoader(test_dataset, batch_size=5,
                                 num_workers=4, shuffle=True, pin_memory=False)

        imgs, msks = next(iter(test_loader))
        plot_batch(imgs, msks, size=5)
        imgs = imgs.to(device, dtype=torch.float)
        msks = msks.to(device, dtype=torch.float)

        preds = []
        for fold in range(20):
            model = load_model('/home/ubuntu/team3/Project_team3/best_model_Kanishk.pt')
            with torch.no_grad():
                pred = model(imgs)
                pred = nn.Sigmoid()(pred)
            #preds.append(pred)

        print(imgs.shape)
        print(type(imgs))
        imgs = imgs.cpu().detach()
        #mean_preds = torch.mean(torch.stack(preds, dim=0), dim=0).cpu().detach()

        plot_batch(imgs, pred, size=5)
        gc.collect()
