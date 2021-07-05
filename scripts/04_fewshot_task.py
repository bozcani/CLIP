import os
import clip
import torch

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm

import torchvision 

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)


def get_features(dataset):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)


    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()



train_dataset = torchvision.datasets.ImageFolder(root="/home/ilker/repos/helsing/coco_crops_few_shot/train", transform=preprocess)
test_dataset = torchvision.datasets.ImageFolder(root="/home/ilker/repos/helsing/coco_crops_few_shot/test", transform=preprocess)

# Calculate the image features
train_features, train_labels = get_features(train_dataset)
print(train_features.shape)
print(train_labels)

test_features, test_labels = get_features(test_dataset)
# Perform logistic regression
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.
print(f"Accuracy = {accuracy:.3f}")
