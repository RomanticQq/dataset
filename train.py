from config import opt
from data.dataset import Dataset
from torch.utils import data as data_
import fire

def train():
    dataset = Dataset(opt)
    dataloader = data_.DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=1)
    for ii, (train_img, val_img) in enumerate(dataloader):
        print(train_img.shape, val_img.shape)
        if ii == 1:
            return


if __name__ == '__main__':

    fire.Fire()
