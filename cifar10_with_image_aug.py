import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

import numpy as np
from PIL import Image

from collections import Counter
from sklearn.model_selection import train_test_split

class CIFAR10_SUB(CIFAR10):
  def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
    super(CIFAR10_SUB, self).__init__(root, train, transform, target_transform,
                 download)
    
  def _under_sample(self, sample_ratio, random_state=42):
    """
    Undersample data according to a ratio. 
    The sampling is stratified (the ratio of class distribution is maintained). 
    """
    self.data, _, self.targets, _ = train_test_split(self.data, self.targets, 
                                                     train_size=sample_ratio, 
                                                     random_state=random_state,
                                                     stratify = self.targets)
    
  def augment(self, augmentations, under_sample_ratio=None, aug_n_times=1):
    """
    under_sample_ratio: float between 0. and 1.
      Select a subset of data with a percentage defined by under_sample_ratio. 
    augmentations: augmentation operations defined by torch.transform.
      If it is None, then skip augmentation. 
    aug_n_times: int. 
      Upsample the dataset by apply augmentations aug_n_times repeatedly. 

    """
    if under_sample_ratio is not None:
      self._under_sample(under_sample_ratio)
    
    if augmentations is None:
      return
    
    n_images = self.data.shape[0]
    augmented = []
    targets = []
    for i in range(n_images):
      img = Image.fromarray(self.data[i])
      for k in range(aug_n_times):
        augmented.append(augmentations(img))
        targets.append(self.targets[i])
    
    self.data = torch.stack(augmented, dim=0)
    self.targets = targets

  def __getitem__(self, index):
    image, target = self.data[index], self.targets[index]
    if isinstance(image, np.ndarray):
      image = Image.fromarray(image)

    return image, target

  def get_data(self):
    return self.data, self.targets

  def get_class_percentage(self):
    label_cnt = Counter(self.targets)
    pct = {}
    for label in label_cnt:
      pct[label] = label_cnt[label] / len(self.targets)
    return pct
    
def test_CIFAR10_SUB():
  # Step 1. Initiate a dataset object:
  subset = CIFAR10_SUB(root='/content/drive/My Drive/cifar10/data', train=True, download=True)

  Xtrain, ytrain = subset.get_data()
  print("Before transformation: ")
  print("X_train.shape = ", Xtrain.shape)
  print("y_train type:", type(ytrain))
  print("Length of y_train", len(ytrain))

  transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])
  # Step 2. Apply augmentation: 
  subset.augment(augmentations=transform_train, under_sample_ratio=0.2, aug_n_times=2)

  Xtrain, ytrain = subset.get_data()
  print()
  print("After transformations:")
  print("len(subset) =", len(subset))
  print("X_train.shape =", Xtrain.shape)
  print("len(ytrain) =", len(ytrain))

  # Step 3. Generate a DataLoader with the augmented dataset: 
  trainloader = torch.utils.data.DataLoader(subset, batch_size=4,
                                          shuffle=False, num_workers=2)
  
  return
