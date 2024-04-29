from colorama import Fore, Back, Style
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
train_bs = 64
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

class FCNN(nn.Module):

    def __init__(self):
        super(FCNN, self).__init__()
        # Learnable layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        nn.init.kaiming_normal(self.conv1.weight)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        nn.init.kaiming_normal(self.conv2.weight)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1)
        nn.init.kaiming_normal(self.conv3.weight)
        # self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, padding=2)
        # nn.init.kaiming_normal(self.conv4.weight)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=5, padding=2)
        nn.init.kaiming_normal(self.conv5.weight)
        self.conv6 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, padding=2)
        nn.init.kaiming_normal(self.conv6.weight)
        # self.conv7 = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=5, padding=2)
        # nn.init.kaiming_normal(self.conv7.weight)
        # self.conv8 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, padding=2)
        # nn.init.kaiming_normal(self.conv8.weight)

        self.upsample_mode = 'nearest'

    # to experiment with other upsampling techniques
    def set_upsample_mode(self, upsample_mode='nearest'):
        if (upsample_mode in ['nearest', 'linear', 'bilinear', 'trilinear']):
            self.upsample_mode = upsample_mode


    def forward(self, x):
        # x.size() = (N, 3, W, W)
        x = F.relu(self.conv1(x))
        # x.size() = (N, 16, W, W)
        x = F.relu(self.conv2(x))
        # x.size() = (N, 32, W, W)
        x = F.max_pool2d(x, (2,2))
        # x.size() = (N, 32, W/2, W/2)
        x = F.relu(self.conv3(x))
        # x.size() = (N, 16, W/2, W/2)
        x = F.upsample(x, scale_factor=2, mode=self.upsample_mode)
        # x.size() = (N, 16, W, W)
        # x = self.conv4(x)
        # x.size() = (N, 2, W, W)
        x = self.conv5(x)
        x = self.conv6(x)
        # x = self.conv7(x)
        # x = self.conv8(x)

        return x

def model_definition():
    model = FCNN()
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
        msk = msks[idx,].permute((1, 2, 0)).numpy()
        show_img(img, msk)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    df_train, valid_ids = read_data()
    test_dataset = BuildDataset(df_train[df_train.index.isin(valid_ids)],)
    test_loader = DataLoader(test_dataset, batch_size=5,
                             num_workers=4, shuffle=False, pin_memory=True)

    imgs, msks = next(iter(test_loader))

    imgs = imgs.to(device, dtype=torch.float)

    preds = []
    for fold in range(1):
        model = load_model('/home/ubuntu/team3/Project_team3/final_model_Kanishk.pt')
        with torch.no_grad():
            pred = model(imgs)
            pred = (nn.Sigmoid()(pred) > 0.5).double()
        preds.append(pred)

    imgs = imgs.cpu().detach()
    preds = torch.mean(torch.stack(preds, dim=0), dim=0).cpu().detach()
    plot_batch(imgs, preds, size=5)
    gc.collect()
