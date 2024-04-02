import os
import argparse
import multiprocessing
from pathlib import Path
from PIL import Image
import zipfile
import numpy as np

import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

from byol_pytorch import BYOL

resnet = models.resnet50(pretrained=False)

parser = argparse.ArgumentParser(description='byol-lightning-test')

parser.add_argument('--image_folder', type=str, required = True,
                       help='path to your folder of images for self-supervised learning')

args = parser.parse_args()

BATCH_SIZE = 128
EPOCHS     = 3
LR         = 3e-4
NUM_GPUS   = 1
IMAGE_SIZE = 32
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
NUM_WORKERS = multiprocessing.cpu_count()
SCL = False

learner = BYOL(
    resnet,
    scl=SCL,
    batch_size=BATCH_SIZE,
    pool_kernel_size=8,
    image_size = IMAGE_SIZE,
    hidden_layer = 'avgpool'
)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

class ImagesDataset(Dataset):
    def __init__(self, folder, image_size):
        super().__init__()
        self._zipfile = zipfile.ZipFile(folder)
        self.folder = self._zipfile.namelist()
        self.paths = []
        class_names = []

        for path in sorted(self.folder):
            _, ext = os.path.splitext(path)
            if ext.lower() in IMAGE_EXTS:
                self.paths.append(path)
            class_names.append(os.path.basename(path).split("_")[0])

        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        self.classes = [sorted_classes[x] for x in class_names]

        print(f'{len(self.paths)} images found')

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        out_dict = {}
        with self._zipfile.open(path, "r") as f:
            img = Image.open(f)
            img.load()
        img = img.convert('RGB')
        out_dict["y"] = np.array(self.classes[index], dtype=np.int64)
        return self.transform(img), out_dict
    

ds = ImagesDataset(args.image_folder, IMAGE_SIZE)
train_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
train_loader = iter(train_loader)

for i in range(EPOCHS):
    images, classes = next(train_loader)
    loss = learner(images)
    opt.zero_grad()
    loss.backward()
    opt.step()
    learner.update_moving_average() # update moving average of target encoder
    print(loss.item())

# save your improved network
torch.save(resnet.state_dict(), './improved-net.pt')