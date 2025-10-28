import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import time  

from model import Net
from data_utils import get_mnist_data, distribute_data
from client import Client
from server import Server

def run_simulation(num_rounds=50, num_clients=100, C=0.1, E=5, B=32, data_dist='non-iid', learning_rate=0.01):
    """
    Runs a single federated learning simulation.
    """
    print(f"Starting simulation: Rounds={num_rounds}, Clients={num_clients}, C={C}, E={E}, B={B}, Dist={data_dist}, LR={learning_rate}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, test_dataset = get_mnist_data()
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    client_datasets = distribute_data(train_dataset, num_clients, iid=(data_dist == 'iid'))

    global_model = Net().to(device)
    server = Server(global_model, test_loader, device)
    
    clients = []
    for i in range(num_clients):
        client_loader = DataLoader(client_datasets[i], batch_size=B, shuffle=True)
        
        client = Client(
            client_id=i,
            local_data_loader=client_loader,
            device=device,
            learning_rate=learning_rate
        )
        clients.append(client)

    accuracy_history = []
    for round_num in range(1, num_rounds + 1):
        
        server.train_communication_round(
            all_clients=clients, 
            fraction_c=C, 
            local_epochs=E
        )
        
        test_loss, test_accuracy = server.evaluate()
        accuracy_history.append(test_accuracy)
        
        if round_num % 5 == 0 or round_num == 1:
            print(f"Round {round_num:3d}/{num_rounds} | Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")

    print("Simulation finished.")
    return accuracy_history

if __name__ == "__main__":
    print("Running a single test simulation (Non-IID Baseline)...")
    history = run_simulation(
        num_rounds=50,
        num_clients=100,
        C=0.1,
        E=5,
        B=32,
        data_dist='non-iid'
    )
    print(f"\nFinal accuracy from test run: {history[-1]:.2f}%")