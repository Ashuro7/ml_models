import torch
import torch.nn as nn
import torchvision.datasets as dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import transforms
import torch.optim as optim

#Defining Hyperparameters
num_epochs = 4
learn_rate = 0.001
num_input = 28 * 28
num_class = 10

#Data loading and transformation
img_transforms = transforms.Compose([transforms.ToTensor()])

train = dataset.MNIST(root='./datasets', train=True, download=True, transform=img_transforms)
test = dataset.MNIST(root='./datasets', train=False, download=True, transform=img_transforms)

train_data = DataLoader(train, batch_size=64, shuffle=True)
test_data = DataLoader(test, batch_size=64, shuffle=True)

#Model architecture definition
class MNISTNet(nn.Module):
    def __init__(self, inp_s, out_s):
        super().__init__()
        self.layer1 = nn.Linear(inp_s, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 64)
        self.layer4 = nn.Linear(64, out_s)
        
        self.neuralnet = nn.Sequential(self.layer1, nn.ReLU(), self.layer2, nn.ReLU(), self.layer3,
                                      nn.ReLU(), self.layer4)
    def forward(self, X):
        output = self.neuralnet(X)
        return output
    
model = MNISTNet(num_input, num_class)

loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=learn_rate)

#Training the model
for ep in range(1, num_epochs+1):
    for idx, data in enumerate(train_data):
        X, y = data
        
        out = model(X.view(-1, 28*28))
        loss = loss_fn(out, y)
        
        loss.backward()
        opt.step()
        
        opt.zero_grad()
        
    print(f'Loss at epoch {ep} was : {loss:.4f}')

#Testing
total = 0
correct = 0

with torch.no_grad():
    for idx, data in enumerate(test_data):
        X,y = data
        output = model(X.view(-1, 28*28))
        idx_max = torch.argmax(output, dim=1)
        for idx, out in enumerate(idx_max):
            total += 1
            if out == y[idx]:
                correct += 1
                
acc = correct/total
print(f'Accuracy is: {acc:.3f}')

#Saving the parameters of the trained model
model_path = 'mnist.pt'

if acc > 0.90:
    torch.save(model.state_dict(), model_path)
    print('Model saved to ' + model_path)