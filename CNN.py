'''
Zeinab Drameh
ACAD 222, Spring 2026
drameh@usc.edu
Final Project Part 2
'''

import torch
from torchvision.io import read_image
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

image_path = './chest_xray'

# ImageFolder loads images from subfolders and assigns labels automatically
# chest_xray/train/NORMAL    -> label 0
# chest_xray/train/PNEUMONIA -> label 1
xray_train_dataset = torchvision.datasets.ImageFolder(image_path + '/train')
xray_valid_dataset = torchvision.datasets.ImageFolder(image_path + '/val')
xray_test_dataset  = torchvision.datasets.ImageFolder(image_path + '/test')

print('Train set:', len(xray_train_dataset))
print('Validation set:', len(xray_valid_dataset))
print('Test set:', len(xray_test_dataset))


#First transformation: cropping to a bounding box
fig = plt.figure(figsize=(16, 8.5))
'''
ax = fig.add_subplot(2, 5, 1)
img, label = xray_train_dataset[0]
ax.set_title('Crop to a \nbounding-box', size=15)
ax.imshow(img, cmap='gray')
ax = fig.add_subplot(2, 5, 6)
img_cropped = transforms.functional.crop(img, 15, 15, 1500, 1500)
ax.imshow(img_cropped, cmap='gray')
plt.show()
'''

'''
## Column 2: flipping (horizontally)
ax = fig.add_subplot(2, 5, 2)
img, label = xray_train_dataset[1]
ax.imshow(img, cmap='gray')
ax = fig.add_subplot(2, 5, 7)
img_flipped = transforms.functional.hflip(img)
ax.imshow(img_flipped, cmap='gray')
plt.show()
'''

'''
## Column 3: adjust contrast
ax = fig.add_subplot(2, 5, 3)
img, label = xray_train_dataset[2]
ax.set_title('Adjust contrast', size=15)
ax.imshow(img, cmap='gray')
ax = fig.add_subplot(2, 5, 8)
img_adj_contrast = transforms.functional.adjust_contrast(
     img, contrast_factor=2)
ax.imshow(img_adj_contrast, cmap='gray')
plt.show()
'''

'''
## Column 4: adjust brightness
ax = fig.add_subplot(2, 5, 4)
img, label = xray_train_dataset[3]
ax.set_title('Adjust brightness', size=15)
ax.imshow(img, cmap='gray')
ax = fig.add_subplot(2, 5, 9)
img_adj_brightness = transforms.functional.adjust_brightness(
     img, brightness_factor=1.3)
ax.imshow(img_adj_brightness, cmap='gray')
plt.show()
'''

'''
## Column 5: cropping from image center
ax = fig.add_subplot(2, 5, 5)
img, label = xray_train_dataset[4]
ax.set_title('Center crop\nand resize', size=15)
ax.imshow(img, cmap='gray')
ax = fig.add_subplot(2, 5, 10)
img_center_crop = transforms.functional.center_crop(img, [0.7*1500, 0.7*1500])
img_resized = transforms.functional.resize(img_center_crop, size=(224, 224))
ax.imshow(img_resized, cmap='gray')
plt.show()
'''

'''
fig = plt.figure(figsize=(22, 12))
print(len(xray_train_dataset))
for i, (img, label) in enumerate(xray_train_dataset):
    ax = fig.add_subplot(3, 6, i*6+1)
    ax.imshow(img, cmap='gray')
    if i == 0:
        ax.set_title('Orig.', size=15)

    ax = fig.add_subplot(3, 6, i*6+2)
    img_transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomCrop([178, 178]),
    ])
    img_cropped = img_transform(img)
    ax.imshow(img_cropped, cmap='gray')
    if i == 0:
        ax.set_title('Step 1: Random crop', size=15)

    ax = fig.add_subplot(3, 6, i*6+3)
    img_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=[178, 178], scale=(0.8, 1.0)),
    ])
    img_resized_crop = img_transform(img_cropped)
    ax.imshow(img_resized_crop, cmap='gray')
    if i == 0:
        ax.set_title('Step 2: Random resize', size=15)

    ax = fig.add_subplot(3, 6, i*6+4)
    img_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
    ])
    img_flip = img_transform(img_resized_crop)
    ax.imshow(img_flip, cmap='gray')
    if i == 0:
        ax.set_title('Step 3: Random flip', size=15)

    ax = fig.add_subplot(3, 6, i*6+5)
    img_transform = transforms.Compose([
        transforms.ColorJitter(brightness=(0.5, 1.5)),
    ])
    img_jitter = img_transform(img_flip)
    ax.imshow(img_jitter, cmap='gray')
    if i == 0:
        ax.set_title('Step 4: Brightness', size=15)

    ax = fig.add_subplot(3, 6, i*6+6)
    img_final = transforms.functional.resize(img_jitter, size=(64, 64))
    ax.imshow(img_final, cmap='gray')
    if i == 0:
        ax.set_title('Step 5: Resize', size=15)
    if i == 2:
        break
plt.show()
'''

# ImageFolder already assigns 0=NORMAL, 1=PNEUMONIA from subfolder names
# No lambda needed - equivalent to get_smile in the CelebA example

#Random augmentation code for our training data
transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize([256, 256]),
        transforms.RandomCrop([178, 178]),
        transforms.RandomResizedCrop(size=[178, 178], scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.Resize([64, 64]),
        transforms.ToTensor(), ])

#Code for our validation and test set transforms
transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize([256, 256]),
        transforms.CenterCrop([178, 178]),
        transforms.Resize([64, 64]),
        transforms.ToTensor(), ])


xray_train_dataset = torchvision.datasets.ImageFolder(
        image_path + '/train',
        transform=transform_train )

torch.manual_seed(1)
data_loader = DataLoader(xray_train_dataset, batch_size=2)
fig = plt.figure(figsize=(15, 6))
num_epochs = 5
for j in range(num_epochs):
    img_batch, label_batch = next(iter(data_loader))
    img = img_batch[0]
    ax = fig.add_subplot(2, 5, j + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Epoch {j}:', size=15)
    ax.imshow(img.permute(1, 2, 0), cmap='gray')

    img = img_batch[1]
    ax = fig.add_subplot(2, 5, j + 6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(img.permute(1, 2, 0), cmap='gray')
plt.show()
'''

'''
#Apply our transformations to the validation and test sets
xray_valid_dataset = torchvision.datasets.ImageFolder(
        image_path + '/val',
        transform=transform )
xray_test_dataset = torchvision.datasets.ImageFolder(
        image_path + '/test',
        transform=transform )

# The Kaggle chest x-ray dataset already comes in a reasonable size
# but we subset to keep training times manageable, similar to CelebA
from torch.utils.data import Subset
xray_train_dataset = Subset(xray_train_dataset, torch.arange(2000))
xray_valid_dataset = Subset(xray_valid_dataset, torch.arange(16))
print('Train set:', len(xray_train_dataset))
print('Validation set:', len(xray_valid_dataset))

#Now create 3 data loaders for our data sets
batch_size = 32
torch.manual_seed(1)
train_dl = DataLoader(xray_train_dataset, batch_size, shuffle=True)
valid_dl = DataLoader(xray_valid_dataset, batch_size, shuffle=False)
test_dl  = DataLoader(xray_test_dataset,  batch_size, shuffle=False)


model = nn.Sequential()
# Input is 1 channel (grayscale) instead of 3 (RGB) - only change from ch14_part2
model.add_module('conv1', nn.Conv2d(
     in_channels=1, out_channels=32, kernel_size=3, padding=1))
model.add_module('relu1', nn.ReLU())
model.add_module('pool1', nn.MaxPool2d(kernel_size=2))

model.add_module('conv2', nn.Conv2d(
     in_channels=32, out_channels=64, kernel_size=3, padding=1))
model.add_module('relu2', nn.ReLU())
model.add_module('pool2', nn.MaxPool2d(kernel_size=2))

model.add_module('conv3', nn.Conv2d(
     in_channels=64, out_channels=128, kernel_size=3, padding=1))
model.add_module('relu3', nn.ReLU())
model.add_module('pool3', nn.MaxPool2d(kernel_size=2))

model.add_module('conv4', nn.Conv2d(
     in_channels=128, out_channels=256, kernel_size=3, padding=1))
model.add_module('relu4', nn.ReLU())


model.add_module('pool4', nn.AvgPool2d(kernel_size=8))
model.add_module('flatten', nn.Flatten())
x = torch.ones((4, 1, 64, 64))
model(x).shape
torch.Size([4, 256])

model.add_module('fc', nn.Linear(256, 1))
model.add_module('sigmoid', nn.Sigmoid())
x = torch.ones((4, 1, 64, 64))
model(x).shape
torch.Size([4, 1])


#device = torch.device("cuda:0")
device = torch.device("cpu")
model = model.to(device)

#define our loss function
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01 )


def train(model, num_epochs, train_dl, valid_dl):
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_dl:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)[:, 0]
            loss = loss_fn(pred, y_batch.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch] += loss.item() * y_batch.size(0)
            is_correct = ((pred >= 0.5).float() == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.sum().cpu()

        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)[:, 0]
                loss = loss_fn(pred, y_batch.float())
                loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
                is_correct = ((pred >= 0.5).float() == y_batch).float()
                accuracy_hist_valid[epoch] += is_correct.sum().cpu()

        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)

        print(
            f'Epoch {epoch + 1} accuracy: {accuracy_hist_train[epoch]:.4f} val_accuracy: {accuracy_hist_valid[epoch]:.4f}')
    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid


torch.manual_seed(1)
num_epochs = 27
hist = train(model, num_epochs, train_dl, valid_dl)
avg_val_acc = np.mean(hist[3])
best_val_acc = np.max(hist[3])

print(f'Average validation accuracy: {avg_val_acc:.4f}')
print(f'Best validation accuracy: {best_val_acc:.4f}')

x_arr = np.arange(len(hist[0])) + 1
fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, hist[0], '-o', label='Train loss')
ax.plot(x_arr, hist[1], '--<', label='Validation loss')
ax.legend(fontsize=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, hist[2], '-o', label='Train acc.')
ax.plot(x_arr, hist[3], '--<', label='Validation acc.')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Accuracy', size=15)

plt.show()


'''
#We can run our trained model on our hold-out test set
accuracy_test = 0
model.eval()
with torch.no_grad():
    for x_batch, y_batch in test_dl:
        pred = model(x_batch)[:, 0]
        is_correct = ((pred>=0.5).float() == y_batch).float()
        accuracy_test += is_correct.sum()
accuracy_test /= len(test_dl.dataset)
print(f'Test accuracy: {accuracy_test:.4f}')


pred = model(x_batch)[:, 0] * 100
fig = plt.figure(figsize=(15, 7))
for j in range(10, 20):
    ax = fig.add_subplot(2, 5, j-10+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(x_batch[j].permute(1, 2, 0), cmap='gray')
    if y_batch[j] == 1:
        label='Pneumonia'
    else:
        label = 'Normal'
    ax.text(0.5, -0.15, f'GT: {label}\nPr(Pneumonia)={pred[j]:.0f}%',
          size=16, horizontalalignment='center', verticalalignment='center',
          transform=ax.transAxes)
plt.show()
'''