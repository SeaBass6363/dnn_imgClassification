import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os



def ddp_setup():
    init_process_group(backend='gloo')

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

class Trainer:
    def __init__(
        self,
        train_data: DataLoader,
        model: nn.Module,
        optimizer: optim.Optimizer,
        save_every: int,
        snapshot_path: str, 
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.train_data = train_data
        self.model = model
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model)

    def _load_snapshot(self, snapshot_path):
        # loc = f"Rank:{self.local_rank}"
        if self.global_rank == 0:
          snapshot = torch.load(snapshot_path)
          self.model.load_state_dict(snapshot["MODEL_STATE"])
          self.epochs_run = snapshot["EPOCHS_RUN"]
          print(f"Resuming training from snapshot at Epoch {self.epochs_run}")


    def _run_batch(self, inputs, labels):
        self.optimizer.zero_grad()
        output = self.model(inputs)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[RANK{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for inputs, labels in self.train_data:
            # inputs = inputs.to(self.local_rank)
            # labels = labels.to(self.local_rank)
            self._run_batch(inputs, labels)

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")


    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.global_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)



def load_train_objs():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)  # load your dataset
    model = Net()  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    return train_set, model, optimizer


def prepare_dataloader(dataset: datasets, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        # num_workers=2,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(total_epochs: int, save_every: int, batch_size: int, snapshot_path: str = "snapshot.pt"):
    ddp_setup()
    train_set, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(train_set, batch_size)
    trainer = Trainer(train_data, model, optimizer, save_every, snapshot_path)
    trainer.train(total_epochs)
    destroy_process_group()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    # parser.add_argument('--batch_size', default=4, type=int, help='Input batch size on each default: 32)')
    args = parser.parse_args()
    
    main(args.total_epochs, args.save_every, 4)