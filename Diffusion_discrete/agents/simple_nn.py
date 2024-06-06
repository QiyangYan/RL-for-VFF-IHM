import torch
import torch.nn as nn
import torch.optim as optim

# Define the Neural Network
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes, mid_layer=128, out_layer=64):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, mid_layer)  # First fully connected layer
        self.relu = nn.ReLU()                  # ReLU Activation
        self.fc2 = nn.Linear(mid_layer, out_layer)          # Second fully connected layer
        self.fc3 = nn.Linear(out_layer, num_classes)  # Output layer

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

    def loss(self, predict, target):
        return self.criterion(predict, target.long())  # Assuming the correct class is 0


if __name__ == '__main__':
    # Parameters
    # obs: 24
    # deisred_goal: 9
    # last action: 1
    # discrete terms: 4
    input_size = 10  # Number of input features (change according to your data)
    num_classes = 4  # Four output classes

    # Model initialization
    model = SimpleNN(input_size, num_classes)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()  # Suitable for classification with N classes
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Common choice for optimization

    # Example forward pass (Assuming you have data in a tensor `inputs`)
    inputs = torch.randn(1, input_size)  # Randomly generated input
    outputs = model(inputs)
    loss = criterion(outputs, torch.tensor([0]))  # Assuming the correct class is 0
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights