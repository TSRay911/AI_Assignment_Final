import torch
import torch.nn as nn
import torch.nn.functional as F

class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()

        self.user_factors = torch.nn.Embedding(n_users,n_factors)
        self.item_factors = torch.nn.Embedding(n_items,n_factors)

        self.user_factors.weight.data.uniform_(0,0.05)
        self.item_factors.weight.data.uniform_(0,0.05)

    def forward(self,data):
        users, items = data[:,0],data[:,1]
        return (self.user_factors(users) * self.item_factors(items)).sum(1)
    
    def predict(self,user,item):
        return self.forward(user,item)
    

class NeuralNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 12, 5) # (12, 220, 220)
        self.pool = nn.MaxPool2d(2, 2) # (12, 110, 110)
        self.conv2 = nn.Conv2d(12, 24, 5) # (24, 106, 106) -> (24, 53, 53)

        self.fc1 = nn.Linear(24 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))         
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



