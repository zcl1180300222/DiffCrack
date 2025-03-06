import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import cv2
import imgaug.augmenters as iaa
from Perlin import rand_perlin_2d_np
import torch

texture_list = ['road']

class BaseDataset(Dataset):

    def __init__(self, img_size):
        self.img_size = img_size[::-1]  # 转换为 (width, height)
        self.normalize = lambda x: x.astype(np.float32) / 255.0

    def _load_image(self, path):
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        return cv2.resize(image, self.img_size)

    def _transform(self, image, mask=None):
        image = self.normalize(image)
        mask = np.zeros_like(image[..., [0]]) if mask is None else self.normalize(mask)
        return image.transpose(2, 0, 1), mask[..., None].transpose(2, 0, 1)


class TestDataset(BaseDataset):

    def __init__(self, data_path, img_size):
        super().__init__(img_size)
        self.root = Path(data_path) / 'test'
        self.image_paths = sorted(self.root.glob('*/*.png'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = Path(self.image_paths[idx])
        is_good = img_path.parent.name == 'good'

        image = self._load_image(img_path)
        mask = None if is_good else self._load_mask(img_path)
        image, mask = self._transform(image, mask)

        return {
            'image': image,
            'mask': mask,
            'has_anomaly': np.float32(not is_good),
            'file_name': f"{img_path.parent.name}_{img_path.name}"
        }

    def _load_mask(self, img_path):
        mask_path = img_path.parents[2] / 'ground_truth' / img_path.parent.name / f"{img_path.stem}_mask.png"
        return cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)


class TrainDataset(BaseDataset):

    def __init__(self, data_path, img_size, args):
        super().__init__(img_size)
        self.root = Path(data_path)
        self.args = args
        self._init_paths()
        self._init_augmenters()

    def _init_paths(self):
        self.image_paths = sorted((self.root / 'train' / 'good').glob('*.png'))
        anomaly_root = Path(self.args["anomaly_source_path"])
        self.anomaly_paths = sorted(anomaly_root.glob('images/*/*.*'))
        self.thresh_paths = sorted((Path(self.args["root_path"]) / 'road' / 'thresh').glob('*.png'))

    def _init_augmenters(self):
        self.base_aug = iaa.SomeOf(1, [
            iaa.pillike.EnhanceSharpness(),
            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
            iaa.Affine(rotate=(-45, 45))
        ])

        self.anomaly_aug = iaa.SomeOf(2, [
            iaa.GammaContrast((0.5, 1.0), per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
            iaa.Solarize(0.5, threshold=(32, 128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, _):
        img_path = np.random.choice(self.image_paths)
        image = self._process_image(img_path)
        return self._create_sample(*self._generate_anomaly(image, img_path))

    def _process_image(self, path):
        image = self._load_image(path)
        return self.normalize(image).transpose(2, 0, 1)

    def _generate_anomaly(self, image, img_path):
        thresh = self._load_thresh(img_path)
        perlin_mask = self._generate_perlin_mask(thresh)

        if np.random.rand() > 0.99 or perlin_mask.sum() == 0:
            return image, np.zeros_like(thresh), 0.0

        anomaly_img = self._apply_anomaly(perlin_mask, image, thresh)
        blended = self._blend_images(image, anomaly_img, perlin_mask)
        return blended, perlin_mask, 1.0

    def _load_thresh(self, img_path):
        if self.classname in texture_list:
            path = np.random.choice(self.thresh_paths)
        else:
            path = Path(str(img_path).replace('train', 'DISthresh'))
        return cv2.resize(cv2.imread(str(path), 0), self.img_size)

    def _generate_perlin_mask(self, thresh):
        for _ in range(50):  # 最多尝试50次
            scale = 2 ** np.random.randint(0, 5, 2)
            noise = rand_perlin_2d_np(self.img_size[::-1], scale)
            mask = (noise > np.random.uniform(0.3, 0.6)).astype(np.float32)
            if mask.sum() > 0:
                return mask * thresh
        return np.zeros_like(thresh)

    def _apply_anomaly(self, mask, image, thresh):
        if self.classname in texture_list or np.random.rand() > 0.5:
            anomaly_img = self.anomaly_aug(image=self._load_random_anomaly())
        else:
            anomaly_img = self._create_patch_anomaly(image)
        return anomaly_img * mask[..., None]

    def _blend_images(self, orig, anomaly, mask):
        beta = np.random.uniform(0, 0.3)
        return orig * (1 - mask) + (1 - beta) * anomaly + beta * orig * mask

    def _load_random_anomaly(self):
        path = np.random.choice(self.anomaly_paths)
        return self.anomaly_aug(image=self._load_image(path))

    def _create_patch_anomaly(self, image):
        shuffled = self._shuffle_patches(image.transpose(1, 2, 0))
        return self.base_aug(image=shuffled).transpose(2, 0, 1)

    def _shuffle_patches(self, img):
        h, w = img.shape[:2]
        grid = [np.split(row, 8, axis=1) for row in np.split(img, 8)]
        return np.vstack([np.hstack(np.random.permutation(row)) for row in grid])