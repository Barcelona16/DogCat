from data import DogCat
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

from torchsummary import summary


if __name__ == '__main__':
    print('hello')
    dataset = DogCat('train', True, 1)
    trainloader = DataLoader(dataset,
                             batch_size=256,
                             shuffle=True,
                             num_workers=12)

    import model
    train_model = getattr(model, 'resnet34')()

    summary(train_model, (3, 224, 224), batch_size=256, device="cpu")

    for ii, (data, label) in enumerate(trainloader):
        print(label)

    print(dataset.imgs)
