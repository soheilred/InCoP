import torch
import sys
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import constants as C

# from torchvision.transforms import ToTensor

class Data:
    def __init__(self, batch_size, data_dir, dataset, transform=None):
        "Load the training DataLoader and the test DataLoader"
        self.batch_size = batch_size
        self.train_kwargs = {'batch_size': batch_size}
        self.test_kwargs = {'batch_size': batch_size}

        if torch.cuda.is_available():
            cuda_kwargs = {'num_workers': 1,
                           'pin_memory': True,
                           'shuffle': True}
            self.train_kwargs.update(cuda_kwargs)
            self.test_kwargs.update(cuda_kwargs)

        if transform is None:
            self.transform = transforms.Compose([
                    # transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    # transforms.Resize(256),
                    # transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010]),
                    # transforms.Normalize(mean=(0.5), std=(0.5)),
                ])

        else:
            self.transform = transform

        # preparing the training and test dataset
        if dataset == "CIFAR10":
            training_data = datasets.CIFAR10(
                root=data_dir,
                train=True,
                download=True,
                transform=self.transform
                )

            test_data = datasets.CIFAR10(
                root=data_dir,
                train=False,
                download=True,
                transform=self.transform
                )

        elif dataset == "CIFAR100":
            training_data = datasets.CIFAR100(
                root=data_dir,
                train=True,
                download=True,
                transform=self.transform
                )

            test_data = datasets.CIFAR100(
                root=data_dir,
                train=False,
                download=True,
                transform=self.transform
                )

        elif dataset == "IMAGENET":
            data_dir = C.DATA_DIR + 'tiny-imagenet-200' 
            train_dir = os.path.join(data_dir, 'train')
            test_dir = os.path.join(data_dir, 'test')

            self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                ])

            training_data = datasets.ImageFolder(train_dir,
                                                transform=self.transform)
            test_data = datasets.ImageFolder(test_dir, transform=self.transform)

        elif dataset == "MNIST":
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            training_data = datasets.MNIST(
                root=data_dir,
                train=True,
                download=True,
                transform=self.transform
                )

            test_data = datasets.MNIST(
                root=data_dir,
                train=False,
                download=True,
                transform=self.transform
                )

        elif dataset == "FashionMNIST":
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            training_data = datasets.FashionMNIST(
                root=data_dir,
                train=True,
                download=True,
                transform=self.transform
                )

            test_data = datasets.FashionMNIST(
                root=data_dir,
                train=False,
                download=True,
                transform=self.transform
                )

        else:
            sys.exit("Wrong dataset name")

        self.num_classes = len(training_data.classes)
        self.train_dataloader = DataLoader(training_data, **self.train_kwargs)
        self.test_dataloader = DataLoader(test_data, **self.test_kwargs)

    def get_num_classes(self):
        return self.num_classes
