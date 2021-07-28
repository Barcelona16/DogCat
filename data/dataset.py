import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T

class DogCat(data.Dataset):
    def __init__(self, state = 'train', transforms = True, k = 0):
        '''
        get paths
        state = train / val / test
        '''
        self.state = state
        self.transforms = transforms
        if self.state != 'test':
            imgs_path = [os.path.join('../dogs-vs-cats/train',img) for img in os.listdir('../dogs-vs-cats/train')]
        else:
            imgs_path = [os.path.join('../dogs-vs-cats/test',img) for img in os.listdir('../dogs-vs-cats/test')]
        print(imgs_path)


    def __getitem__(self, item):
        return self.imgs

    def __len__(self):
        return len(self.imgs)

