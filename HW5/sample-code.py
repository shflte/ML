import csv
import cv2
import numpy as np
import random
import os

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

TRAIN_PATH = "train"
TEST_PATH = "test"
device = "cpu"
# try device = "cuda" 
# and change your settings/accelerator to GPU if you want it to run faster

class Task1Dataset(Dataset):
    def __init__(self, data, root, return_filename=False):
        self.data = [sample for sample in data if sample[0].startswith("task1")]
        self.return_filename = return_filename
        self.root = root
    
    def __getitem__(self, index):
        filename, label = self.data[index]
        img = cv2.imread(f"{self.root}/{filename}")
        img = cv2.resize(img, (32, 32))
        img = np.mean(img, axis=2)
        if self.return_filename:
            return torch.FloatTensor((img - 128) / 128), filename
        else:
            return torch.FloatTensor((img - 128) / 128), int(label)

    def __len__(self):
        return len(self.data)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 10)
        )
        
        
    def forward(self, x):
        b, h, w = x.shape
        x = x.view(b, h*w)
        return self.layers(x)


train_data = []
val_data = []

with open(f'{TRAIN_PATH}/annotations.csv', newline='') as csvfile:
    for row in csv.reader(csvfile, delimiter=','):
        if random.random() < 0.7:
            train_data.append(row)
        else:
            val_data.append(row)

train_ds = Task1Dataset(train_data, root=TRAIN_PATH)
train_dl = DataLoader(train_ds, batch_size=500, num_workers=4, drop_last=True, shuffle=True)

val_ds = Task1Dataset(val_data, root=TRAIN_PATH)
val_dl = DataLoader(val_ds, batch_size=500, num_workers=4, drop_last=False, shuffle=False)


train_data = []
val_data = []

with open(f'{TRAIN_PATH}/annotations.csv', newline='') as csvfile:
    for row in csv.reader(csvfile, delimiter=','):
        if random.random() < 0.7:
            train_data.append(row)
        else:
            val_data.append(row)

train_ds = Task1Dataset(train_data, root=TRAIN_PATH)
train_dl = DataLoader(train_ds, batch_size=500, num_workers=4, drop_last=True, shuffle=True)

val_ds = Task1Dataset(val_data, root=TRAIN_PATH)
val_dl = DataLoader(val_ds, batch_size=500, num_workers=4, drop_last=False, shuffle=False)

test_data = []
with open(f'{TEST_PATH}/../sample_submission.csv', newline='') as csvfile:
    for row in csv.reader(csvfile, delimiter=','):
        test_data.append(row)

test_ds = Task1Dataset(test_data, root=TEST_PATH, return_filename=True)
test_dl = DataLoader(test_ds, batch_size=500, num_workers=4, drop_last=False, shuffle=False)

if os.path.exists('submission.csv'):
    csv_writer = csv.writer(open('submission.csv', 'a', newline=''))
else:
    csv_writer = csv.writer(open('submission.csv', 'w', newline=''))
    csv_writer.writerow(["filename", "label"])

model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()


for epoch in range(100):
    print(f"Epoch [{epoch}]")
    model.train()
    for image, label in train_dl:
        image = image.to(device)
        label = label.to(device)
        
        pred = model(image)
        loss = loss_fn(pred, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    sample_count = 0
    correct_count = 0
    model.eval()
    for image, label in val_dl:
        image = image.to(device)
        label = label.to(device)
        
        pred = model(image)
        loss = loss_fn(pred, label)
        
        pred = torch.argmax(pred, dim=1)
        
        sample_count += len(image)
        correct_count += (label == pred).sum()
        
    print("accuracy (validation):", correct_count / sample_count)


model.eval()
for image, filenames in test_dl:
    image = image.to(device)
    
    pred = model(image)
    pred = torch.argmax(pred, dim=1)
    
    for i in range(len(filenames)):
        csv_writer.writerow([filenames[i], str(pred[i].item())])

for filename, _ in test_data:
    if filename.startswith("task2") or filename.startswith("task3"):
        csv_writer.writerow([filename, 0])
    

