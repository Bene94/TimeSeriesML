import torch
import torch.nn as nn
import torch.optim as optim
from models import NanoGPT
from dataloader import TimeSeriesDataset

class NNTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.criterion = nn.MSELoss()  # or choose an appropriate loss function
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def train(self, train_loader, epochs=10):
        self.model.train()
        for epoch in range(epochs):
            for time_series, const, operation, target in train_loader:
                # Assuming your model can handle these inputs and has three outputs
                outputs = self.model(time_series, const, operation)

                # Calculate loss here, assuming target is a tuple of three elements
                loss = coustom_loss(outputs, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")

    def evaluate(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            for time_series, const, operation, target in test_loader:
                outputs = self.model(time_series, const, operation)
                loss = sum([self.criterion(out, tgt) for out, tgt in zip(outputs, target)])
                # Here you can add code to calculate accuracy or other metrics
                print(f"Test Loss: {loss.item()}")

    def coustom_loss(output, target):
        # if target[:,3] = inf than mask the rest
        output[torch.isinf(target[:,3]),0] = 0
        output[torch.isinf(target[:,3]),1] = 0

        loss_0 = torch.mean((output[:,0] - target[:,0])**2)
        loss_1 = torch.mean((output[:,1] - target[:,1])**2)
        loss_2 = torch.mean((torch.sigmoid(output[:,1]) - target[:,1])**2) 

        return loss_0 + loss_1 + loss_2 

# Example usage
model = NanoGPT()  # Initialize your model here
trainer = NNTrainer(model)
train_dataset = TimeSeriesDataset(data_path="path/to/your/data")
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
trainer.train(train_loader)
test_dataset = TimeSeriesDataset(data_path="path/to/your/data")
test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
trainer.evaluate(test_loader)

