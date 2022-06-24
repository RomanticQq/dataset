import os
import numpy as np
from PIL import Image


class VOCBboxDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.dir_list = os.listdir(self.data_dir)

    def __len__(self):
        return len(self.dir_list)

    def get_example(self, i):
        img_file = os.path.join(self.data_dir, self.dir_list[i])
        img = read_image(img_file, color=True)
        return img

    # __getitem__ = get_example


def read_image(path, dtype=np.float32, color=True):
    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))