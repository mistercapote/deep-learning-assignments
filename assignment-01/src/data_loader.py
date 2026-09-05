import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
from pathlib import Path


def generate_image(img_size: int, seed: int = None):
    image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed=seed)
    num_ellipses = int(rng.integers(5, 21))
    instance_masks = np.zeros((num_ellipses, img_size, img_size), dtype= np.uint8)
    
    for idx in range(num_ellipses):
        center = tuple(rng.integers(0, img_size + 1, size=2).tolist())
        axes = tuple(rng.integers(5, 41, size=2).tolist())
        angle = int(rng.integers(0, 101))
        gray_intensity = int(rng.integers(50, 256))
        color = (gray_intensity, gray_intensity, gray_intensity)
        
        cv.ellipse(image, center, axes, angle, 0, 360, color, -1)
        cv.ellipse(instance_masks[idx], center, axes, angle, 0, 360, 1, -1)
        
    contrast = rng.random() + 0.5
    noise = rng.integers(-20, 20, size=(img_size, img_size, 3))
    image = (image * contrast) + noise
    image = np.clip(image, 0 , 255 ).astype(np.uint8)

    return image, instance_masks


def to_tensors(image: np.ndarray, instance_masks: np.ndarray):
    image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0

    binary_mask = (instance_masks.sum(axis=0) > 0).astype(np.float32)
    binary_mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0)

    instance_gt = np.zeros(instance_masks.shape[1:], dtype=np.int32)
    for i, mask in enumerate(instance_masks):
        instance_gt[mask > 0] = i + 1
        
    return image_tensor, binary_mask_tensor, instance_gt


class SyntheticEllipseDataset(Dataset):
    def __init__(self, num_samples: int, img_size: int = 128, seed: int = None) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.img_size = img_size
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seed = None if self.seed is None else self.seed + idx
        image, instance_masks = generate_image(self.img_size, seed=seed)            
        return to_tensors(image, instance_masks)


class DSB2018Dataset(Dataset):
    def __init__(self, root_dir: str, img_size: int = 128, sample_ids: list = None):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.sample_ids = sample_ids or sorted(
            p.name for p in self.root_dir.iterdir() if p.is_dir()
        )

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_dir = self.root_dir / self.sample_ids[idx]

        image_path = next((sample_dir / "images").glob("*.png"))
        image = cv.imread(str(image_path), cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (self.img_size, self.img_size), interpolation=cv.INTER_LINEAR)

        mask_paths = sorted((sample_dir / "masks").glob("*.png"))
        instance_masks = np.zeros((len(mask_paths), self.img_size, self.img_size), dtype=np.uint8)
        for i, mask_path in enumerate(mask_paths):
            mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)
            mask = cv.resize(mask, (self.img_size, self.img_size), interpolation=cv.INTER_NEAREST)
            instance_masks[i] = (mask > 127).astype(np.uint8)

        return to_tensors(image, instance_masks)


def build_dataset(dataset_type: str, **kwargs):
    if dataset_type == "synthetic":
        return SyntheticEllipseDataset(**kwargs)
    elif dataset_type == "real":
        return DSB2018Dataset(**kwargs)