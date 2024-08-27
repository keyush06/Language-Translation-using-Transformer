import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple deep network
class DeepNetwork(nn.Module):
    def __init__(self):
        super(DeepNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        return self.layers(x)

# Initialize the network, loss function, and optimizer
model = DeepNetwork()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Generate dummy data
input_data = torch.randn(64, 100)
target_data = torch.randn(64, 1)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(input_data)
    loss = criterion(outputs, target_data)
    loss.backward()
    
    # Check for exploding gradients
    for param in model.parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2)
            if grad_norm > 1e3:  # Example threshold
                print(f"Exploding gradient detected at epoch {epoch}: {grad_norm}")

    optimizer.step()
