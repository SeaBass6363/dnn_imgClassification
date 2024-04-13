import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Load the saved model
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
    

def test(rank, size):
    #Model Setup to evaluation mode
    model = Net()
    model.module.load_state_dict(torch.load('trained_model.pth'))
    model.eval() 

    #Loading the trained model
    # if rank ==0:
    #     model.module.load_state_dict(torch.load('trained_model.pth'))
    
    # Define test data transformations
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load test dataset
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the network on the test images: {accuracy:.2f}%")

if __name__ == '__main__':
    test()

# loaded_model = Net()

# loaded_model.module.load_state_dict(torch.load('trained_model.pth'))
# loaded_model.eval()


# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# trainset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=size, rank=rank)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=False, num_workers=2, sampler=train_sampler)



# # Load and preprocess the unseen image
# image_path = 'test.jpg'  # Replace with the path to your image
# image = Image.open(image_path)
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
# input_tensor = preprocess(image)
# input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension


# # Perform inference
# with torch.no_grad():
#     output = model(input_batch)

# # Get the predicted class
# _, predicted_class = output.max(1)

# # Map the predicted class to the class name
# class_names = ['daisy', 'dandelion']  # Make sure these class names match your training data
# predicted_class_name = class_names[predicted_class.item()]

# print(f'The predicted class is: {predicted_class_name}')


# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image

# # Display the image with the predicted class name
# image = np.array(image)
# plt.imshow(image)
# plt.axis('off')
# plt.text(10, 10, f'Predicted: {predicted_class_name}', fontsize=12, color='white', backgroundcolor='red')
# plt.show()