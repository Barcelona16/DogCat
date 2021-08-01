from data import DogCat
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

if __name__ == '__main__':
    print('hello')
    dataset = DogCat('train', True, 1)
    trainloader = DataLoader(dataset,
                             batch_size=256,
                             shuffle=True,
                             num_workers=12)
    
    for ii, (data, label) in enumerate(trainloader):
        print(label)

    print(dataset.imgs)
