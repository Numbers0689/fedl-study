import torch
import torch.nn as nn
import random
import copy

class Server:
    def __init__(self, global_model, test_loader, device):
        self.global_model = global_model.to(device)
        self.test_loader = test_loader
        self.device = device

        self.criterion = nn.NLLLoss(reduction='sum')

    def select_clients(self, all_clients, fraction_c):
        """
        Selects a fraction C of all clients.
        :param all_clients: List of all Client objects.
        :param fraction_c: Fraction of clients to select (C).
        :return: List of selected Client objects.
        """
        num_selected = max(1, int(len(all_clients) * fraction_c))
        selected_clients = random.sample(all_clients, num_selected)
        return selected_clients

    def aggregate_updates(self, client_updates, total_data_samples):
        """
        Aggregates client model updates using Federated Averaging (FedAvg).
        :param client_updates: A list of (state_dict, num_samples) tuples.
        :param total_data_samples: Total data points across all *selected* clients.
        """
        aggregated_weights = copy.deepcopy(client_updates[0][0])
        
        for key in aggregated_weights:
            aggregated_weights[key] = torch.zeros_like(aggregated_weights[key])

        for key in aggregated_weights:
            for i in range(len(client_updates)):
                client_weights, num_samples = client_updates[i]
                weight_fraction = num_samples / total_data_samples
                aggregated_weights[key] += client_weights[key] * weight_fraction
        
        self.global_model.load_state_dict(aggregated_weights)

    def train_communication_round(self, all_clients, fraction_c, local_epochs):
        """
        Runs one complete communication round.
        1. Select clients
        2. Send global model to clients
        3. Clients train locally
        4. Collect and aggregate updates
        """
        selected_clients = self.select_clients(all_clients, fraction_c)
        
        client_updates = []
        total_data_samples = 0
        
        global_weights = self.global_model.state_dict()
        for client in selected_clients:
            client.set_model_weights(global_weights)
            
            num_samples = client.train(local_epochs=local_epochs)
            
            client_updates.append((client.get_model_weights(), num_samples))
            total_data_samples += num_samples
            
        if client_updates:
            self.aggregate_updates(client_updates, total_data_samples)
            
        return len(selected_clients)

    def evaluate(self):
        """Evaluates the global model on the test set."""
        self.global_model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                output = self.global_model(images)

                test_loss += self.criterion(output, labels).item()
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        
        return test_loss, accuracy