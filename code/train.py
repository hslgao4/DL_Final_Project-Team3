from libraries import *

'''Set up random seed'''
np.random.seed(6303)
random.seed(6303)
torch.manual_seed(6303)
torch.cuda.manual_seed(6303)

'''Hyper-parameters'''
batch_size = 100
num_workers = 4
image_size = 224
epoch = 1
LR = 0.0002

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_csv('../code/final_df.csv')
print(df.shape)   # (38496, 11)

# Remove images with no masks
dff = df.copy()
df_train = dff.dropna(subset=df.iloc[:, 1:4].columns, how='all')
print(df_train.shape)
df_train.fillna('', inplace=True)

# Split into train， validation and test
train_df, temp_test_df = train_test_split(df_train, test_size=0.2, random_state=6303)
valid_df, test_df = train_test_split(temp_test_df, test_size=0.5, random_state=6303)



'''Dataloader and set dataset'''
class CustomDataset(Dataset):

    def __init__(self, df, subset='train', augmentation=None):
        self.df = df
        self.subset = subset
        self.augmentation = augmentation

    def __getitem__(self, index):
        path = self.df.path.iloc[index]
        width = self.df.width.iloc[index]
        height = self.df.height.iloc[index]
        image = self.load_image(path)
        # if self.subset == 'train':
        # generate mask
        masks = np.zeros((image_size, image_size, 3), dtype=np.float32)
        for i, j in enumerate(["large_bowel", "small_bowel", "stomach"]):
            rles = self.df[j].iloc[index]
            mask = rle_decode(rles, shape=(height, width, 1))
            mask = cv2.resize(mask, (image_size, image_size))
            masks[:, :, i] = mask

        masks = masks.transpose(2, 0, 1)   # change dimension order to (channel, height, width)
        image = image.transpose(2, 0, 1)

        return (torch.tensor(image), torch.tensor(masks))   # if self.subset == 'train' else torch.tensor(image)

    def __len__(self):
        return len(self.df)

    def load_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        image = cv2.resize(image, (image_size, image_size))
        image = np.expand_dims(image, -1)
        return image.astype(np.float32) / 255

data_augmentation = {
    "train": A.Compose([A.Resize(image_size, image_size, interpolation=cv2.INTER_NEAREST),
                        A.HorizontalFlip(),
                        A.VerticalFlip()], p=1.0),

    "valid": A.Compose([A.Resize(image_size, image_size, interpolation=cv2.INTER_NEAREST), ], p=1.0),

    "test": A.Compose([A.Resize(image_size, image_size, interpolation=cv2.INTER_NEAREST), ], p=1.0)
}

train_data = CustomDataset(train_df, augmentation=data_augmentation['train'])
valid_data = CustomDataset(valid_df, augmentation=data_augmentation['valid'])
test_data = CustomDataset(valid_df, augmentation=data_augmentation['test'])

params_1 = {
    'batch_size': batch_size,
    'shuffle': True,
    'num_workers': num_workers
}

params_2 = {
    'batch_size': batch_size,
    'shuffle': False,
    'num_workers': num_workers
}

train = DataLoader(train_data, **params_1)
valid = DataLoader(valid_data, **params_2)
test = DataLoader(test_data, **params_2)

# check size
image, mask = next(iter(train))
print(image.size())
print(mask.size())

# Mask visualization - sample
def plot_image_mask(image, mask, n=5):
    plt.figure(figsize=(5*5, 5))
    for i in range(n):
        plt.subplot(1, n, i+1)
        images = image[i, ].permute((1, 2, 0)).numpy()
        masks = mask[i, ].permute((1, 2, 0)).numpy()
        show_img(images, masks)
    plt.tight_layout()
    plt.show()

plot_image_mask(image, mask, n=5)



''' U-Net Model '''
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.D_conv = DoubleConv(in_channels, 64)

        self.Down1 = Down(64, 128)
        self.Down2 = Down(128, 256)
        self.Down3 = Down(256, 512)
        self.Down4 = Down(512, 512)

        self.Up1 = Up(512, 256)
        self.Up2 = Up(256, 128)
        self.Up3 = Up(128, 64)
        self.Up4 = Up(64, 64)

        self.Out_Conv = OutConv(64, out_channels)

    def forward(self, x):
        d0 = self.D_conv(x)
        d1 = self.Down1(d0)
        d2 = self.Down2(d1)
        d3 = self.Down3(d2)
        d4 = self.Down4(d3)

        u1 = self.Up1(d4, d3)
        u2 = self.Up2(u1, d2)
        u3 = self.Up3(u2, d1)
        u4 = self.Up4(u3, d0)

        logits = self.Out_Conv(u4)

        return logits

def UNET():
    model = UNet(in_channels=1, out_channels=3)
    model.to(device)
    return model

model = UNET()
criterion = monai.losses.DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)

for epoch in range(epoch):
    loss = []
    for i, (image, mask) in enumerate(train):
        images = Variable(image).to(device)
        masks = Variable(mask).to(device)
        optimizer.zero_grad()
        prediction = model(images)
        loss = criterion(prediction, masks)
        loss.backward()
        optimizer.step()
        loss.append(loss.item())
    print(f'Epoch : {epoch} --> Loss: {sum(loss)/len(loss)}')



