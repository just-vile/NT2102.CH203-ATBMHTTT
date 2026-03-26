# fl_mnist_demo/server_app.py
import logging
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Context, ndarrays_to_parameters
from fl_mnist_demo.task import MNISTNet, get_weights

logger = logging.getLogger(__name__)

def server_fn(context: Context):
    # Initialize global model
    model = MNISTNet()
    initial_params = ndarrays_to_parameters(get_weights(model))

    strategy = FedAvg(
        fraction_fit=1.0,          # Use 100% of available clients
        fraction_evaluate=1.0,
        min_fit_clients=2,         # Wait for both clients
        min_evaluate_clients=2,
        min_available_clients=2,   # Don't start until 2 clients connected
        initial_parameters=initial_params,
    )

    config = ServerConfig(num_rounds=5)
    return strategy, config

# Register the ServerApp
app = ServerApp(server_fn=server_fn)