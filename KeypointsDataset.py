from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from ground_truth import generate_gt
from face_alignment import face_alignment
import matplotlib.image as mpimg
import cv2
import os
import torch


class KeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                self.key_pts_frame.iloc[idx, 0])

        image = mpimg.imread(image_name)

        # if image has an alpha color channel, get rid of it
        if(image.shape[2] == 4):
            image = image[:,:,0:3]

        key_pts = self.key_pts_frame.iloc[idx, 1:].to_numpy()
        key_pts = key_pts.astype('float').reshape(-1, 2)

        image, key_pts = face_alignment(image, key_pts)

        ground_truth = generate_trainable_gt(key_pts)

        sample = {'image_name': image_name, 'image': image, 'keypoints': key_pts, 'ground_truth': ground_truth}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image_name, image, key_pts, ground_truth = sample['image_name'], sample['image'], sample['keypoints'], sample['ground_truth']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))

        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

        # re-predict ground_truth
        ground_truth = generate_trainable_gt(key_pts)

        return {'image_name': image_name, 'image': torch.from_numpy(img).float().view(3, 64, 64), 'keypoints': key_pts, 'ground_truth': torch.from_numpy(ground_truth).float()}


def generate_trainable_gt(key_pts):
    gt_list = generate_gt(key_pts)

    trainable_gt = []
    for idx in range(len(gt_list)):
        if idx < 6:
            trainable_gt.append(gt_list[idx])
        else:
            trainable_gt += gt_list[idx]
    return np.array(trainable_gt)


face_dataset = KeypointsDataset(csv_file='ytb/training_frames_keypoints.csv',
                                root_dir='ytb/training/',
                                transform=Rescale((64, 64)))


print("Number of data: ", len(face_dataset))
print("Ground Truth Size: ", face_dataset[1]['ground_truth'].shape)
print("Image Size: ", face_dataset[1]['image'].shape)
