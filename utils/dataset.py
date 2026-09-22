import glob
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

AIRWAY_LABELS = ['V_airway', 'O_airway', 'OTE']


def crop_roi(image, mask):
    if image.shape[:2] == (480, 720):
        image = image[30:414, 254:670]
        mask = mask[30:414, 254:670]
    elif image.shape[:2] == (480, 640):
        image = image[30:414, 234:586]
        mask = mask[30:414, 234:586]
    return image, mask


def points_to_mask(image_shape, points):
    mask = np.zeros(image_shape, dtype=np.uint8)
    if not points:
        return mask
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    return mask


class Rescale():
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        img = sample['image']
        label = sample.get('label', None)

        if len(img.shape) < 3:
            img = np.reshape(img, (img.shape[0], img.shape[1], 1))

        h, w = img.shape[:2]
        channel = img.shape[-1]
        scale = min(self.output_size / w, self.output_size / h)
        new_h, new_w = int(scale * h), int(scale * w)
        image_resized = cv2.resize(img, (new_w, new_h))
        pad_h, pad_w = (self.output_size - new_h) // 2, (self.output_size - new_w) // 2

        image_paded = np.zeros([self.output_size, self.output_size, channel], dtype=np.uint8)
        image_paded[pad_h:new_h + pad_h, pad_w:new_w + pad_w, :] = image_resized
        sample['image'] = image_paded

        if label is not None:
            label_resized = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            label_paded = np.zeros([self.output_size, self.output_size, 1], dtype=np.uint8)
            label_paded[pad_h:new_h + pad_h, pad_w:new_w + pad_w, :] = np.expand_dims(label_resized, axis=-1)
            sample['label'] = label_paded

        return sample


class LoadData(Dataset):
    def __init__(self, data_root_folder, preprocessing=None, transform=None, transform_rate=0.75):
        self.data_root_folder = data_root_folder
        self.preprocessing = preprocessing
        self.transform = transform
        self.transform_rate = transform_rate

        self.image_paths = sorted(glob.glob(os.path.join(data_root_folder, '*/jpg/*.jpg')) +
                                  glob.glob(os.path.join(data_root_folder, '*/jpg/*.png')))
        self.label_paths = sorted(glob.glob(os.path.join(data_root_folder, '*/json/*.json')))
        print(f'{data_root_folder} -> image: {len(self.image_paths)}  label: {len(self.label_paths)}')

    def __len__(self):
        return len(self.image_paths)

    def read_image(self, image_path):
        with open(image_path, 'rb') as f:
            nparr = np.frombuffer(f.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f'Could not read image: {image_path}')
        return image[..., ::-1]

    def read_mask(self, label_path, image_shape):
        with open(label_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)

        mask = np.zeros(image_shape, dtype=np.uint8)
        for shape in label_data['shapes']:
            if shape['label'] in AIRWAY_LABELS:
                points = np.array(shape['points'], dtype=np.int32).reshape(-1, 2).tolist()
                mask[points_to_mask(image_shape, points) > 0] = 255
        return mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.image_paths[idx]
        image = self.read_image(image_path)
        mask = self.read_mask(self.label_paths[idx], image.shape[:2])
        image, mask = crop_roi(image, mask)

        sample = {'image': image, 'label': mask}
        if self.preprocessing:
            sample = self.preprocessing(sample)
        if self.transform and np.random.uniform() < self.transform_rate:
            sample = self.transform(sample)

        return {'image': transforms.ToTensor()(sample['image']),
                'label': transforms.ToTensor()(sample['label']),
                'image_path': image_path}


class build_dataset():
    def __init__(self, image_size, batch_size):
        self.batch_size = batch_size
        self.preprocessing = transforms.Compose([Rescale(image_size)])

    def train(self, data_root_folder, augmentation=None, transform_rate=0.75, num_workers=0):
        dataset = LoadData(data_root_folder, self.preprocessing, augmentation, transform_rate)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=num_workers, drop_last=True)

    def valid(self, data_root_folder):
        dataset = LoadData(data_root_folder, self.preprocessing)
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    def test(self, data_root_folder):
        return self.valid(data_root_folder)
