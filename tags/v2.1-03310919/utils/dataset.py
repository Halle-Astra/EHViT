# dataset.py

import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset

class StereoMatchingDataset(Dataset):
    def __init__(self, data_dir, mode="training",transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        self.left_images = sorted(glob.glob(os.path.join(data_dir,mode,'image_2', '*.png')))
        self.right_images = sorted(glob.glob(os.path.join(data_dir, mode,'image_3', '*.png')))
        self.depth_maps0 = sorted(glob.glob(os.path.join(data_dir, mode,'disp_occ_0', '*.png')))
        self.depth_maps1 = sorted(glob.glob(os.path.join(data_dir, mode,'disp_occ_1', '*.png')))
        self.depth_maps = sorted(self.depth_maps0+self.depth_maps1)
    def __len__(self):
        return len(self.left_images)

    def __getitem__(self, idx):
        try: 
            left_image = Image.open(self.left_images[idx]).convert('L')
            right_image = Image.open(self.right_images[idx]).convert('L')
            depth_map = Image.open(self.depth_maps[idx]).convert('L')
        except IndexError as e:
            print(f"Caught IndexError for idx: {idx}")
            print(f"Left image paths length: {len(self.left_images)}")
            print(f"Right image paths length: {len(self.right_images)}")
            raise e

        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)
            depth_map = self.transform(depth_map)

        if left_image.shape != (1,375,1242):
            dim_1, dim_2, dim_3 =min(1, left_image.shape[0]),min(375, left_image.shape[1]),min(1242, left_image.shape[2])
            mask = torch.zeros((1, 375, 1242))
            mask[:dim_1, :dim_2, :dim_3] = left_image[:dim_1, :dim_2, :dim_3]
            left_image = mask

        if right_image.shape != (1,375,1242):
            dim_1, dim_2, dim_3 =min(1, right_image.shape[0]),min(375, right_image.shape[1]),min(1242, right_image.shape[2])
            mask = torch.zeros((1, 375, 1242))
            mask[:dim_1, :dim_2, :dim_3] = right_image[:dim_1, :dim_2, :dim_3]
            right_image = mask
        
        if depth_map.shape != (1,375,1242):
            dim_1, dim_2, dim_3 =min(1, depth_map.shape[0]),min(375, depth_map.shape[1]),min(1242, depth_map.shape[2])
            mask = torch.zeros((1, 375, 1242))
            mask[:dim_1, :dim_2, :dim_3] = depth_map[:dim_1, :dim_2, :dim_3]
            depth_map = mask
        assert left_image.shape == right_image.shape, f'left != right, idx: {idx}'
        assert left_image.shape == depth_map.shape, f'left != dep, idx: {idx}'
        assert right_image.shape == depth_map.shape, f'dep != right, idx: {idx}'
        return_ls = []
        for img in [left_image, right_image, depth_map]:
            img_ = torch.nn.functional.interpolate(img[:,None, :,:], (180,600))[:,0,:,:]#(90,300))[:,0,:,:]
            return_ls.append(img_)
        left_image, right_image, depth_map = return_ls
        return left_image, right_image, depth_map

