import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T

class DogCat(data.Dataset):
    def __init__(self, state = 'train', transforms = True, k = 1):
        '''
        get paths
        k = 0 1 2 3 4 - k_fold
        state = train / val / test
        k_fold
        self.imgs
        '''
        self.state = state
        self.transforms = transforms
        self.k_fold = 5
        if self.state == 'train':
            imgs_path = [os.path.join('../dogs-vs-cats/train',img) for img in os.listdir('../dogs-vs-cats/train')]
            data_size = len(imgs_path)/self.k_fold
            imgs = sorted(imgs_path, key=lambda x: int(x.split('.')[-2]))

            np.random.seed(613)
            imgs = np.random.permutation(imgs)

            self.imgs = imgs[int(k*data_size):int((k+1)*data_size)]
            # print(len(self.imgs))
            # print(len(imgs_path))

        else:
            imgs_path = [os.path.join('../dogs-vs-cats/test',img) for img in os.listdir('../dogs-vs-cats/test')]
            imgs = sorted(imgs_path, key=lambda x: int(x.split('.')[-2].split('/')[-1]))

            np.random.seed(921)
            imgs = np.random.permutation(imgs)

            self.imgs = imgs

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        if self.state == 'train':
            self.transforms = T.Compose([
                T.Resize(256),
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        elif self.state == 'val' or self.state == 'test':
            self.transforms = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                normalize
            ])

            # print(imgs_path)

    def __getitem__(self, item):
        img_path = self.imgs[item]
        if self.state == 'test':
            label = int(self.imgs[item].split('.')[-2].split('/')[-1])
        elif self.state == 'train' or self.state == 'val':
            label = 1 if 'dog' in img_path.split('/')[-1] else 0

        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)

