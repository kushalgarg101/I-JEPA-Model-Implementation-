import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import torch.nn.functional as TF
import os
from PIL import Image
import numpy as np
import math
from torch.utils.data import Dataset,DataLoader

images_path = r"D:\Image_Jepa\src\utils\photos_no_class"

class ImagePatchesDataset(Dataset):
    """
    loads images from a directory, applies transforms,and splits each image into patches.

    Returns:
        Tensor of shape [patch_dim, num_patches], where patch_dim = C * kernel_size * kernel_size.
    """
    def __init__(self, images_path, kernel_size=16, transform=None):
        super().__init__()

        self.images_path = images_path
        self.image_files = [os.path.join(images_path, f) for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
        self.kernel_size = kernel_size
        self.stride = kernel_size

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('RGB')

        # Apply transform
        img_tensor = self.transform(img)  # shape: [C, H, W]

        # Unfold into patches
        # Output shape: [C * kernel_size * kernel_size, num_patches]
        patches = TF.unfold(img_tensor, kernel_size=self.kernel_size, stride=self.stride)

        return patches




if __name__ == '__main__':
    images_data = ImagePatchesDataset(images_path)
    data_loader = DataLoader(images_data, batch_size= 4, shuffle= True)
    for batch in data_loader:
        print(batch)
        print(batch.size()) # output : [4,768,196]
        break