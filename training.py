import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist, init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import Process

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from config import get_default_config, get_file_path, get_latest_file_path, ModelConfig
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


# Initialize distributed training environment
def init_process(rank, size, fn, backend='gloo'):
    # os.environ['MASTER_ADDR'] = '10.1.1.6' # Replace with IP address of VM that the script is being run on
    # os.environ['MASTER_PORT'] = '12345'  # Choose an available port number (higher than 1024)
    # os.environ['WORLD_SIZE'] = str(size)  # Number of VMs participating in training (so 2)
    # os.environ['RANK'] = str(rank)  # Rank of the current VM (0 or 1)

    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['rank'])

    dist.init_process_group(backend=backend, init_method='env://', rank=rank, world_size=size)
    fn(rank, size)

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train():
    # Set up model, loss function, optimizer, and data loaders
    
    model = Net()
    if os.path.exists('latest_checkpoint.pth'):
        model.load_state_dict(torch.load('latest_checkpoint.pth'))

    model = DDP(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_sampler = DistributedSampler(trainset, shuffle=True)
    trainloader = DataLoader(trainset, shuffle=False, sampler=train_sampler)

    # Training
    for epoch in range(5):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            outputs = model(inputs)

            # if(step_number + 1) % 100 != 0 and not last_step:
            #     with model.no_sync():
            #         loss = criterion(outputs, labels)
            #         loss.backward()
            
           # else: 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f"[{local_rank}, {epoch + 1:3d}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0
        
        if global_rank == 0:
                torch.save(model.state_dict(), 'latest_checkpoint.pth')

    print(f"Finished Training on Rank {global_rank}")

    # Save the trained model
    if rank == 0:  # Save only from one process to avoid multiple saves
        torch.save(model.state_dict(), 'trained_model.pth')

if __name__ == '__main__':

    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['rank'])

    dist.init_process_group(backend='gloo')

    train()

    #destroy_process_group()
    #fn(rank, size)
    