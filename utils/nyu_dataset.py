import json
import os
import random

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from .tof_sim import get_hist_parallel, sample_point_from_hist_parallel, make_tof_config
def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})
def rgb_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    return Image.open(path)

def depth_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)

    return Image.open(path)

def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class NYUv2ToFDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        json_file: str,
        split: str = "train",
        image_size: int = 256,
        depth_min: float = 0.0,
        depth_max: float = 10.0,
        input_height: int = 416,
        input_width: int = 544,
        train_zone_num: int = 8,
        train_zone_random_offset: int = 0,
        zone_sample_num: int = 16,
        sample_uniform: bool = False,
        simu_max_distance: float = 4.0,
        drop_hist: float = 0.0,
        noise_mean: float = 0.0,
        noise_sigma: float = 0.0,
        noise_prob: float = 0.0,
        normalize_rgb: bool = True,
        transform: bool = True,
    ):
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.normalize_rgb = normalize_rgb
        self.K_list = torch.Tensor([
            5.1885790117450188e+02,
            5.1946961112127485e+02,
            3.2558244941119034e+02 - 26.0,
            2.5373616633400465e+02 - 16.0
        ])


        self.cfg = make_tof_config(
            mode="train" if split == "train" else "test",
            input_height=input_height,
            input_width=input_width,
            train_zone_num=train_zone_num,
            train_zone_random_offset=train_zone_random_offset,
            zone_sample_num=zone_sample_num,
            sample_uniform=sample_uniform,
            simu_max_distance=simu_max_distance,
            drop_hist=drop_hist,
            noise_mean=noise_mean,
            noise_sigma=noise_sigma,
            noise_prob=noise_prob,
            do_random_rotate=True if split == "train" else False,
            transform = transform,
        )

        self.data_root = data_root
        

        with open(json_file, 'r') as f:
            print(json_file)
            json_data = json.load(f)
            self.sample_list = json_data[split]

        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.sample_list)

    def _normalize_depth(self, depth: torch.Tensor) -> torch.Tensor:
        depth = depth.clamp(min=self.depth_min, max=self.depth_max)
        depth = (depth - self.depth_min) / max(self.depth_max - self.depth_min, 1e-6)
        depth = depth * 2.0 - 1.0
        return depth

    def __getitem__(self, idx: int):
        focal = self.K_list[0].item()

        rel_path = self.sample_list[idx]["filename"]
        if self.cfg.mode == "train":
            path_file = os.path.join(self.data_root,'/'.join(self.sample_list[idx]['filename'].split    ('/')[1:]))
            # path_file = os.path.join(self.data_root, rel_path)
            num = path_file.split('/')[-1].split('.')[0]
            rgb_path = os.path.join('/'.join(path_file.split('/')[:-1]),'rgb_{}.jpg'.format(num))
            depth_path = os.path.join('/'.join(path_file.split('/')[:-1]),'sync_depth_{}.png'.format(num))
        else:
            path_file = os.path.join(self.data_root,'/'.join(self.sample_list[idx]['filename'].split('/')[1:]))
            num = path_file.split('/')[-1].split('.')[0]
            rgb_path = os.path.join('/'.join(path_file.split('/')[:-1]),'rgb_{}.jpg'.format(num))
            depth_path = os.path.join('/'.join(path_file.split('/')[:-1]),'sync_depth_{}.png'.format(num))
        


        rgb = rgb_loader(rgb_path)
        depth_gt = depth_loader(depth_path)

        # random crop to input_height/input_width for training
        if self.cfg.mode == "train":
            depth_gt = depth_gt.crop((26, 16, 640-26, 480-16))
            rgb = rgb.crop((26, 16, 640-26, 480-16))
            if self.cfg.do_random_rotate is True:
                random_angle = (random.random() - 0.5) * 2 * self.cfg.degree
                rgb = self.rotate_image(rgb, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)

            rgb = np.array(rgb) / 255.0
            depth_gt = np.array(depth_gt, dtype=np.float32) / 1000.0 ## Image read as mm.
            depth_gt = np.expand_dims(depth_gt, axis=2)

            rgb, depth_gt = self.train_preprocess(rgb, depth_gt)
            rgb = np.array(rgb, dtype=np.float32)
            sample = {'image': rgb, 'depth': depth_gt, 'focal': focal}


        else:
            depth_gt = depth_gt.crop((26, 16, 640-26, 480-16))
            rgb = rgb.crop((26, 16, 640-26, 480-16))
            rgb = np.array(rgb, dtype=np.float32) / 255.0
            depth_gt = np.array(depth_gt, dtype=np.float32) / 1000.0 ## Image read as mm.
            depth_gt = np.expand_dims(depth_gt, axis=2)
            
            sample = {'image': rgb, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': True}
            
        

        if self.cfg.transform:
            sample = transforms.Compose([ToTensor(mode=self.cfg.mode)])(sample)

        # simulate ToF from resized tensors
        hist_data, fr, mask = get_hist_parallel(rgb, depth_gt, self.cfg)
        if self.split == "train" and self.cfg.drop_hist > 1e-3:
            index = np.where(mask == True)[0]
            index = np.random.choice(index, int(len(index) * self.cfg.drop_hist))
            mask[index] = False
        if self.split == "train" and self.cfg.noise_prob > 1e-3:
            prob = np.random.random(hist_data[mask, 0].shape)
            noise_mask = prob < self.cfg.noise_prob
            noise = np.random.normal(self.cfg.noise_mean, self.cfg.noise_sigma, hist_data[mask, 0].shape)
            hist_data[mask, 0][noise_mask] += noise[noise_mask]

        fh = sample_point_from_hist_parallel(hist_data, mask, self.cfg)
        sample['hist_data'] = fh.to(torch.float)
        sample['rect_data'] = fr.to(torch.float)
        sample['mask'] = mask.to(torch.bool)

        return sample
    
    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth
    
    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt
    
    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        brightness = random.uniform(0.75, 1.25)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug
    
class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        # import ipdb; ipdb.set_trace()
        image = self.normalize(image)

        # if self.mode == 'test':
        #     return {'image': image, 'focal': focal}

        depth = sample['depth']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            return {'image': image, 'depth': depth, 'focal': focal}
        else:
            has_valid_depth = sample['has_valid_depth']
            depth = self.to_tensor(depth)
            return {'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            # img = torch.from_numpy(pic.copy().transpose((2, 0, 1)))
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img