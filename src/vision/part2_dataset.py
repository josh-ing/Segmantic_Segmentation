#!/usr/bin/python3

import os
import os.path
from typing import List, Tuple

import cv2
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset


"""
Dataset class for semantic segmentation data.
"""

def make_dataset(split: str, data_root: str, data_list_fpath: str) -> List[Tuple[str, str]]:
    """
    Create list of (image file path, label file path) pairs, as ordered in the
    data_list_fpath .txt file.

    Args:
        split: string representing split of data set to use, must be either
            'train','val','test'
        data_root: path to where data lives, and where relative image paths are
            relative to
        data_list_fpath: path to .txt file with relative image paths and their
            corresponding GT path

    Returns:
        image_label_list: list of 2-tuples, each 2-tuple is comprised of an absolute image path
            and an absolute label path
    """
    assert split in ["train", "val", "test"]

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    # print(data_root, data_list_fpath)
    # image_label_list = []
    # for className in os.listdir(data_root):
    #     txt = os.path.join(data_root, className)
    #     for txtFile in os.listdir(txt):
    #         label = os.path.join(data_list_fpath, txtFile)
    #         image_label_list.append((txt, label))
    image_label_list = []
    with open(data_list_fpath) as f:
        lines = f.readlines()
        for l in lines:
            image, label = l.split(' ')
            image1 = image.replace('\n', '')
            label1 = label.replace('\n', '')
            image2 = os.path.join(data_root, image1)
            label2 = os.path.join(data_root, label1)
            image_label_list.append((image2, label2))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    print(f"List of (image,label) pairs {split} list generated!")
    return image_label_list


class SemData(Dataset):
    def __init__(self, split: str, data_root: str, data_list_fpath: str, transform=None) -> None:
        """
        Dataloader class for semantic segmentation datasets.

        Args:
            split: string representing split of data set to use, must be either
                'train','val','test'
            data_root: path to where data lives, and where relative image paths
                are relative to
            data_list_fpath: path to .txt file with relative image paths
            transform: Pytorch torchvision transform
        """
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list_fpath)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the transformed RGB image and ground truth, as tensors.

        We will not load the image using PIL, since we will not be using the
        default Pytorch transforms.

        You can read in the image and label map using imageio or opencv, but
        the transform should accept a (H,W,C) float 32 RGB image (not BGR like
        OpenCV reads), and a (H,W) int64 label map.

        Args:
            index: index of the example to retrieve within the dataset

        Returns:
            image: tensor of shape (C,H,W), with type torch.float32
            label: tensor of shape (H,W), with type torch.long (64-bit integer)
        """

        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################
        imagePath, labelPath = self.data_list[index]
        colorImage = cv2.imread(imagePath, 1)
        label = imageio.imread(labelPath)
        label = label.astype(np.int64)
        image = cv2.cvtColor(colorImage, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        if self.transform != None:
            image, label = self.transform(image = image, label = label)
        # image = torch.tensor(image)
        # label = torch.tensor(label)
        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################
        return image, label
