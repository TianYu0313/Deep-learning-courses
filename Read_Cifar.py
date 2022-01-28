import torchvision
import torch
from torchvision import transforms



def cifar100_dataset(bs):
    CIFAR_PATH = "./cifar100/"
    mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    num_workers = 2

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # 数据增强
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    cifar100_training = torchvision.datasets.CIFAR100(root=CIFAR_PATH, train=True, download=True,
                                                      transform=transform_train)

    cifar100_training, cifar100_validating = torch.utils.data.random_split(cifar100_training, [int(0.9*len(cifar100_training)), int(0.1*len(cifar100_training))])
    trainloader = torch.utils.data.DataLoader(cifar100_training, batch_size=bs, shuffle=True,
                                              num_workers=num_workers)

    validloader = torch.utils.data.DataLoader(cifar100_validating, batch_size=bs, shuffle=True,
                                              num_workers=num_workers)

    cifar100_testing = torchvision.datasets.CIFAR100(root=CIFAR_PATH, train=False, download=True,
                                                     transform=transform_test)
    testloader = torch.utils.data.DataLoader(cifar100_testing, batch_size=bs, shuffle=False, num_workers=num_workers)

    return trainloader, validloader, testloader
