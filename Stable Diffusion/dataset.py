import torch
import torchvision
import numpy as np
from torch.utils.data import Subset


def getDataKMNIST(batch_size, typedata='both', test_size = 10000):


    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.KMNIST(root='./data', train=True, download=True,  transform=transform)
    testset = torchvision.datasets.KMNIST(root='./data', train=False, download=True,  transform=transform)

    assert len(testset) >= test_size, f"testset size cannot be {test_size}, should be less than {len(testset)}! "

    # Split testset 
    I = np.random.permutation(len(testset))
    ds_test = Subset(testset, I[:test_size])
    
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True ,  num_workers=8)
    test_dataloader = torch.utils.data.DataLoader(ds_test, batch_size=batch_size,  num_workers=8)

    if(typedata == 'both'):
        return train_dataloader, test_dataloader
    if(typedata == 'train'):
        return train_dataloader
    if(typedata == 'test'):
        return test_dataloader

def getDataMNIST(batch_size, typedata='both', test_size = 10000):

    
    transform= torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
        
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True,  transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True,  transform=transform)

    assert len(testset) >= test_size, f"testset size cannot be {test_size}, should be less than {len(testset)}! "

    # Split testset 
    I = np.random.permutation(len(testset))
    ds_test = Subset(testset, I[:test_size])
    
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True ,  num_workers=8)
    test_dataloader = torch.utils.data.DataLoader(ds_test, batch_size=batch_size,  num_workers=8)

    if(typedata == 'both'):
        return train_dataloader, test_dataloader
    if(typedata == 'train'):
        return train_dataloader
    if(typedata == 'test'):
        return test_dataloader


def getData(datasetname,batch_size=64, typedata='both', test_size = 10000):

    if(datasetname == 'MNIST'):
        getdataset = getDataMNIST
    if(datasetname == 'KMNIST'):
        getdataset = getDataKMNIST

    
    return getdataset(batch_size, typedata, test_size)
    


    







