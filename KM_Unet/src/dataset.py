import os

import cv2
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

        self.img_ids = []
        for img_id in img_ids:
            img_path = os.path.join(self.img_dir, img_id + self.img_ext)
            if os.path.exists(img_path):
                self.img_ids.append(img_id)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img_path = os.path.join(self.img_dir, img_id + self.img_ext)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f'Image not found: {img_path}')

        mask = []
        for i in range(self.num_classes):
            candidate_paths = [
                os.path.join(self.mask_dir, img_id + self.mask_ext),
                os.path.join(self.mask_dir, img_id + '.png'),
                os.path.join(self.mask_dir, str(i), img_id + self.mask_ext),
                os.path.join(self.mask_dir, str(i), img_id + '.png'),
            ]
            mask_img = None
            for candidate_path in candidate_paths:
                mask_img = cv2.imread(candidate_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    break
            if mask_img is None:
                raise FileNotFoundError(f'Mask not found for {img_id}; tried: {candidate_paths}')
            mask.append(mask_img[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)

        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)

        if mask.max() < 1:
            mask[mask > 0] = 1.0

        img = torch.tensor(img, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return img, mask, {'img_id': img_id}
