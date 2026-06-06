import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class PyTorchAdultIncomeModel(nn.Module):
    """
    PyTorch neural network for Adult Income binary classification.
    """

    def __init__(self, input_size):
        super(PyTorchAdultIncomeModel, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


def prepare_pytorch_data(X_train_processed, X_test_processed, y_train, y_test):
    """
    Convert NumPy/Pandas data into PyTorch tensors and DataLoader.
    """

    X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    return train_loader, X_test_tensor, y_test_tensor


def train_pytorch_model(train_loader, input_dim, epochs=20):
    """
    Train the PyTorch model using a manual training loop.
    """

    torch.manual_seed(42)

    model = PyTorchAdultIncomeModel(input_dim)

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            logits = model(batch_X)
            loss = loss_function(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        train_losses.append(average_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}")

    return model, train_losses


def predict_pytorch_model(model, X_test_tensor):
    """
    Predict class labels using the trained PyTorch model.
    """

    model.eval()

    with torch.no_grad():
        test_logits = model(X_test_tensor)
        test_probabilities = torch.sigmoid(test_logits)
        predictions = (test_probabilities >= 0.5).int().numpy().ravel()

    return predictions, test_probabilities