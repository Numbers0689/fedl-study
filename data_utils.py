import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_mnist_data():
    """
    Downloads and prepares the MNIST dataset.
    
    Returns:
        tuple: (train_dataset, test_dataset) as torch.utils.data.Dataset objects.
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        './data', train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset

def distribute_data(train_dataset, num_clients, iid=True):
    """
    Distributes the training dataset among clients.
    
    Args:
        train_dataset (Dataset): The full training dataset.
        num_clients (int): The number of clients.
        iid (bool): True for IID, False for Non-IID distribution.
        
    Returns:
        list: A list of torch.utils.data.Dataset objects (Subsets) for each client.
    """

    
    client_datasets = []
    
    if iid:
        num_samples_per_client = len(train_dataset) // num_clients
        all_indices = list(range(len(train_dataset)))
        np.random.shuffle(all_indices)
        
        for i in range(num_clients):
            start_idx = i * num_samples_per_client
            end_idx = (i + 1) * num_samples_per_client
            subset_indices = all_indices[start_idx:end_idx]
            client_dataset = Subset(train_dataset, subset_indices)
            client_datasets.append(client_dataset)
    
    else:
        num_shards, num_imgs_per_shard = 200, 300  # 100 clients, 2 shards each
        
        labels = np.array(train_dataset.targets)
        idx_by_class = {i: np.where(labels == i)[0] for i in range(10)}
        
        shard_indices = []
        for i in range(10): # For each class
            class_indices = idx_by_class[i]
            np.random.shuffle(class_indices)
            for j in range(0, len(class_indices), num_imgs_per_shard):
                shard = class_indices[j : j + num_imgs_per_shard]
                shard_indices.append(shard)
        
        np.random.shuffle(shard_indices)
        
        for i in range(num_clients):
            assigned_shards = shard_indices[i * 2 : (i + 1) * 2]
            subset_indices = np.concatenate(assigned_shards)
            client_dataset = Subset(train_dataset, subset_indices)
            client_datasets.append(client_dataset)

    return client_datasets