import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_loaders(dataset, dataroot, batch_size):
    if dataset == 'svhn':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        train_dataset = datasets.SVHN(root=dataroot, split='train', transform=transform, download=True)
        val_dataset = datasets.SVHN(root=dataroot, split='test', transform=transform, download=True)
        test_dataset = datasets.SVHN(root=dataroot, split='test', transform=transform, download=True)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if dataset == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = datasets.CIFAR10(root=dataroot, train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root=dataroot, train=False, download=True, transform=transform)

        # Split trainset into train and validation sets
        train_size = int(0.8 * len(trainset))
        val_size = len(trainset) - train_size
        trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    
    return train_loader, val_loader, test_loader