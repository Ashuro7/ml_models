import torch
import torch.nn as nn
import torchvision.datasets as dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch.optim as optim
from tqdm import tqdm

#Defining Hyperparameters
num_epochs = 4
learn_rate = 0.001
num_input = 28 * 28
num_class = 10
batch_s = 64

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Data loading and transformation
img_transforms = transforms.Compose([transforms.ToTensor()])

train = dataset.MNIST(root='./datasets', train=True, download=True, transform=img_transforms)
test = dataset.MNIST(root='./datasets', train=False, download=True, transform=img_transforms)

train_data = DataLoader(train, batch_size=batch_s, shuffle=True)
test_data = DataLoader(test, batch_size=batch_s, shuffle=True)

#Model architecture definition
class MNISTNet(nn.Module):
    def __init__(self, inp_s, out_s):
        super(MNISTNet, self).__init__()
        self.inp = nn.Linear(inp_s, 64)
        self.hd1 = nn.Linear(64,64)
        self.out = nn.Linear(64, out_s)
        
        self.forward_pass = nn.Sequential(self.inp, nn.ReLU(), self.hd1, nn.ReLU(), 
                                       self.out)
    def forward(self, X):
        output = self.forward_pass(X)
        return output
    
model = MNISTNet(num_input, num_class).to(device)

loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=learn_rate)

#Training the model
for ep in range(1, num_epochs+1):
    for data in tqdm(train_data):
        X, y = data        
        X, y = X.view(-1, 28*28).to(device), y.to(device)
        
        out = model(X)
        loss = loss_fn(out, y)
        
        loss.backward()
        opt.step()
        
        opt.zero_grad(set_to_none=True)
        
    print(f'Epoch: {ep}, Loss: {loss:.4f}')

#Testing
total = 0
correct = 0

with torch.no_grad():
    for idx, data in enumerate(test_data):
        X,y = data
        X,y = X.view(-1, 28*28).to(device), y.to(device)
        
        output = model(X)
        idx_max = torch.argmax(output, dim=1).to(device)
        correct += (idx_max == y).sum().item()
        total += batch_s
                
acc = round(correct/total,4) * 100
print(f'Accuracy of the MNIST Neural Network is: {acc}%')

#Saving the parameters of the trained model
model_path = 'mnist.pt'

if acc > 95.0:
    torch.save(model.state_dict(), model_path)
    print('Model parameters saved to ' + model_path)