import torch
import numpy as np
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch.optim as optim

#Hyper-Parameters
num_epochs = 30
learn_rate = 0.1
b = 128
dataset_path = ''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Data Loading and Transformation
image = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = CIFAR10(root=dataset_path, download=True, train=True, transform=image)
test_data = CIFAR10(root=dataset_path,download=True, train=False, transform=image)

train = DataLoader(dataset=train_data, batch_size=b, shuffle=True)
test = DataLoader(dataset=test_data, batch_size=b, shuffle=True)

print('**Building the model..**')

class CIFAR(nn.Module):
    def __init__(self):
        super(CIFAR, self).__init__()
        
        self.conv1 = nn.Conv2d(3,32,3,padding=4) 
        self.conv2 = nn.Conv2d(32,64,3,padding=4)
        self.conv3 = nn.Conv2d(64,128,3,padding=4)
        
        self.pool = nn.MaxPool2d(2,2) 
        self.ReLU = nn.ReLU(inplace=True) 
        
        self.fc1 = nn.Linear(10368, 5184)
        self.fc2 = nn.Linear(5184,2500)
        self.fc3 = nn.Linear(2500,1000)
        self.fc4 = nn.Linear(1000,10)
        
        self.SequenceConv2d = nn.Sequential(self.conv1, self.ReLU, self.pool, self.conv2, self.ReLU,
                                     self.pool, self.conv3, self.ReLU, self.pool)
        
        self.SequenceLinear = nn.Sequential(self.fc1, self.ReLU, self.fc2, self.ReLU, self.fc3, 
                                           self.ReLU, self.fc4)
        
    def forward(self, x):
        output = self.SequenceConv2d(x)
        output = self.SequenceLinear(output.view(-1, np.prod(output[0].shape)))
        return output

model = CIFAR().to(device)

if str(device) == 'cuda':
    cudnn.benchmark = True
    
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                 mode='min',factor=0.5,patience=1)

print('**Training the Model...**')

for iters in tqdm(range(1,num_epochs+1)):
    for (X,y) in train:
        X, y = X.to(device), y.to(device)
        
        out = model(X)
        loss = loss_fn(out, y)
        
        loss.backward()
        optimizer.step()

        optimizer.zero_grad(set_to_none=True)

    scheduler.step(loss)

    if iters % 5 == 0:
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(f'Epoch: {iters}, loss: {loss:.3f}, learning_rate: {lr:.5f}')

print('**Checking the Accuracy of the model..**')

model.eval()

total = 0
correct = 0

with torch.no_grad():
    for idx, (X, y) in enumerate(test):
        X, y = X.to(device), y.to(device)
        out = model(X)
        
        correct += (torch.argmax(out, dim=1) == y).sum().item()
        total += b
        
    acc = round(correct/total,4) * 100
    print(f'Accuracy of the CNN is: {acc}%')
