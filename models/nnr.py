import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
#------------------
import os
import sys
import numpy as np


sys.path.append('.') # point to the root directory
ROOT_DIR = os.path.dirname(os.path.abspath('COMP_9417_23_Project')) # project Directory
data_dir = os.path.join(ROOT_DIR, 'data')

# read dataset
file = os.path.join(data_dir, 'training.csv')
data_file =  pd.read_csv(file)

#import utility/helper function class
from utils import util
# clean data and drop or replace outliers- set drop_outlers to True to drop or False (default) to replace them
data = util.clean_outliers(data_file, drop_outliers = True )

# set test_split_ratio =0.0 ( X_test and y_test becomes None) when preparing for Final Submission Otherwise set to (0.1 or 0.2): default-0.2
X_train, X_test, y_train, y_test = util.prepare_data(data, test_split_ratio=0.2)

# Convert the data to tensors:
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Create data loaders for training and testing:
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the neural network model:
class DNNRegressor(nn.Module):
    def __init__(self, input_dim, num_layers):
        super(DNNRegressor, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else 128, 128))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 5))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Initialize the model, loss function, and optimizer:
input_dim = X_train.shape[1]
num_layers = 3
learning_rate = 0.001
optimizer_choice = optim.Adam

model = DNNRegressor(input_dim, num_layers)
criterion = nn.MSELoss()
optimizer = optimizer_choice(model.parameters(), lr=learning_rate)

# Train the model:
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y.squeeze())

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # Evaluate the model on test data
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y.squeeze())
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss:.4f}")