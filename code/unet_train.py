from toolbox import *
import pdb


'''Set up random seed'''
np.random.seed(6303)
random.seed(6303)
torch.manual_seed(6303)
torch.cuda.manual_seed(6303)

'''Hyper-parameters'''
batch_size = 64
num_workers = 4
image_size = 224
epoch = 20
LR = 0.0002
SAVE_MODEL = True
model_name = 'UNET_dice'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_csv('../code/final_df.csv')
print(df.shape)   # (38496, 11)

# Remove images with no masks
dff = df.copy()
df_train = dff.dropna(subset=df.iloc[:, 1:4].columns, how='all')
print(df_train.shape)
df_train = df_train.copy()
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
        # image = np.tile(image[..., None], [1, 1, 3])   #RGB
        image = np.expand_dims(image, -1)   #Grey
        return image.astype(np.float32) / 255

data_augmentation = {
    "train": A.Compose([A.Resize(image_size, image_size, interpolation=cv2.INTER_NEAREST),
                        # A.HorizontalFlip(),
                        # A.VerticalFlip()
                        ],
                       p=1.0),

    "valid": A.Compose([A.Resize(image_size, image_size, interpolation=cv2.INTER_NEAREST), ], p=1.0)

}

train_data = CustomDataset(train_df, augmentation=data_augmentation['train'])
valid_data = CustomDataset(valid_df, augmentation=data_augmentation['valid'])
test_data = CustomDataset(test_df, augmentation=data_augmentation['valid'])

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
    plt.figure(figsize=(25, 5))
    for i in range(n):
        plt.subplot(1, n, i+1)
        images = image[i, ].permute((1, 2, 0)).numpy()
        masks = mask[i, ].permute((1, 2, 0)).numpy()
        show_img(images, masks)
    plt.tight_layout()
    plt.show()

# plot_image_mask(image, mask, n=5)



''' U-Net Model '''
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

def UNET():
    model = UNet(in_channels=1, channels=24, out_channels=3)
    model.to(device)
    return model

model = UNET()
DiceLoss = smp.losses.DiceLoss(mode='multilabel')
BCELoss = smp.losses.SoftBCEWithLogitsLoss()
def criterion(y_pred, y_true):
    return DiceLoss(y_pred, y_true)
    # return 0.5*BCELoss(y_pred, y_true) + 0.5*DiceLoss(y_pred, y_true)

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)

train_loss = []
val_loss = []
DICE = []

for epoch in range(epoch):
    Loss_t = []
    Loss_v = []
    dice = []
    model.train()
    for i, (image, mask) in enumerate(tqdm(train, desc=f"Epoch {epoch+1}")):
        images = Variable(image).to(device)
        masks = Variable(mask).to(device)
        optimizer.zero_grad()
        prediction = model(images)
        loss = criterion(prediction, masks)
        loss.backward()
        optimizer.step()
        Loss_t.append(loss.item())

    model.eval()
    with torch.no_grad():
        for image, mask in tqdm(valid, desc=f"Epoch {epoch+1} Validation"):
            images = Variable(image).to(device)
            masks = Variable(mask).to(device)
            prediction = model(images)
            loss = criterion(prediction, masks)
            Loss_v.append(loss.item())
            prediction = (nn.Sigmoid()(prediction) > 0.5).double()
            val_dice = dice_coe(masks, prediction).cpu().detach().numpy()
            dice.append(val_dice)

    train_loss.append(sum(Loss_t) / len(Loss_t))
    val_loss.append(sum(Loss_v) / len(Loss_v))
    DICE.append(sum(dice) / len(dice))
    print(f'Epoch{epoch + 1} --> Train Loss: {sum(Loss_t) / len(Loss_t)}')
    print(f'Epoch{epoch + 1} --> Validation Loss: {sum(Loss_v) / len(Loss_v)}, DICE coe: {sum(dice) / len(dice)}')

    if len(DICE) >= 2:
        if DICE[-1] > DICE[-2] and SAVE_MODEL:
            torch.save(model.state_dict(), "model_{}.pt".format(model_name))


'''Plot the DICE Loss of Train and Validation'''
plt.figure()
plt.plot(np.arange(epoch+1) + 1, train_loss, label='Train loss')
plt.plot(np.arange(epoch+1) + 1, val_loss, label='Validation loss')
plt.xlabel("Epoch")
plt.ylabel("DICE Loss")
plt.title('DICE loss: Train vs. Validation')
plt.xticks(np.arange(epoch+1) + 1)
plt.tight_layout()
plt.legend()
plt.show()

plt.figure()
plt.plot(np.arange(epoch+1) + 1, DICE, label='DICE coe')
plt.title('Validation DICE Coefficient')
plt.xlabel("Epoch")
plt.ylabel("DICE coe")
plt.xticks(np.arange(epoch+1) + 1)
plt.tight_layout()
plt.legend()
plt.show()



'''Test model'''
dice0 = []
dice1 = []
dice2 = []
model = UNET()
model.load_state_dict(torch.load('model_UNET_dice.pt', map_location=device))

with torch.no_grad():
    for image, mask in tqdm(test, desc=f"Epoch {1} Test"):
        images = Variable(image).to(device)
        masks = Variable(mask).to(device)
        prediction = model(images)
        prediction = (nn.Sigmoid()(prediction) > 0.5).double()

        val_dice = dice_coe(masks[:, 0:1], prediction[:, 0:1]).cpu().detach().numpy()
        dice0.append(val_dice)

        val_dice = dice_coe(masks[:, 1:2], prediction[:, 1:2]).cpu().detach().numpy()
        dice1.append(val_dice)

        val_dice = dice_coe(masks[:, 2:3], prediction[:, 2:3]).cpu().detach().numpy()
        dice2.append(val_dice)



print(f'Test DICE coe 0: {sum(dice0) / len(dice0)}')
print(f'Test DICE coe 1: {sum(dice1) / len(dice1)}')
print(f'Test DICE coe 2: {sum(dice2) / len(dice2)}')




'''Visualization'''
image_v, mask_v = next(iter(test))
plot_image_mask(image_v, mask_v, n=5)


pred = []
with torch.no_grad():
    images = Variable(image_v).to(device)
    prediction = model(images)
    prediction = (nn.Sigmoid()(prediction) > 0.5).double()
pred.append(prediction)

pred = torch.mean(torch.stack(pred, dim=0), dim=0).cpu().detach()
plot_image_mask(image_v, pred, n=5)

