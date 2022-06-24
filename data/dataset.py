from __future__ import  absolute_import
from __future__ import  division
import torch as t
from data.voc_dataset import VOCBboxDataset
from skimage import transform as sktsf
from torchvision import transforms



def pytorch_normalze(img):
    normalize = transforms.Compose([
        transforms.CenterCrop(256),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # 一定要记住，这里不能加transforms.ToTensor()
    ])
    img = normalize(t.from_numpy(img))
    return img.numpy()


def preprocess(img, min_size=600, max_size=1000):

    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
    normalize = pytorch_normalze
    return normalize(img)


class Transform(object):

    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img = in_data
        img = preprocess(img, self.min_size, self.max_size)
        return img


class Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.train_db = VOCBboxDataset(opt.train_data_dir)
        self.train_tsf = Transform(opt.min_size, opt.max_size)
        self.val_db = VOCBboxDataset(opt.val_data_dir)
        self.val_tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, idx):
        train_ori_img = self.train_db.get_example(idx)
        train_img = self.train_tsf(train_ori_img)
        val_ori_img = self.val_db.get_example(idx)
        val_img = self.val_tsf(train_ori_img)
        return train_img.copy(), val_img.copy()

    def __len__(self):
        return len(self.train_db)

