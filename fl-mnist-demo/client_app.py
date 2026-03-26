# fl_mnist_demo/client_app.py
import flwr as flwr
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from fl_mnist_demo.task import MNISTNet, load_data, get_weights, set_weights, train, test, DEVICE

class FlowerClient(NumPyClient):
    def __init__(self, partition_id, num_partitions):
        self.model = MNISTNet().to(DEVICE)
        self.train_loader, self.test_loader = load_data(partition_id, num_partitions)

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        train(self.model, self.train_loader, epochs=1)
        return get_weights(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        loss, accuracy = test(self.model, self.test_loader)
        return float(loss), len(self.test_loader.dataset), {'accuracy': float(accuracy)}

def client_fn(context: Context):
    partition_id = context.node_config['partition-id']
    num_partitions = context.node_config['num-partitions']
    return FlowerClient(int(partition_id), int(num_partitions)).to_client()

# Register the ClientApp
app = ClientApp(client_fn=client_fn)
