import torch
import torch.nn as nn
import torch.optim as optim
import copy
from model import Net 

class Client:
    def __init__(self, client_id, local_data_loader, device, learning_rate):
        self.client_id = client_id
        self.local_data_loader = local_data_loader
        self.device = device
        self.learning_rate = learning_rate
        self.model = Net().to(self.device) # Use the imported Net
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        self.criterion = nn.NLLLoss()

    def set_model_weights(self, global_weights):
        """Sets the client's model weights from the global model."""
        self.model.load_state_dict(copy.deepcopy(global_weights))

    def train(self, local_epochs):
        """
        Trains the client's model locally for E epochs.
        :param local_epochs: Number of local epochs (E).
        """
        self.model.train()
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            for images, labels in self.local_data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
        
        return len(self.local_data_loader.dataset)

    def get_model_weights(self):
        """Returns the client's model weights."""
        return self.model.state_dict()