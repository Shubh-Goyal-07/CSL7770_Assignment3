import torch
import torch.nn as nn

from datasets import get_dummy_dataloaders
from model import load_model
from utils import CompressedSpectralLoss

import argparse
import os


EPOCHS = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 32

def train_model(model, train_dataloader, optimizer, criterion, device):
    model.train()

    epoch_loss = 0

    for batch_idx, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output, _ = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        step_loss = loss.item()
        epoch_loss += step_loss

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/ {len(train_dataloader)}, Loss: {step_loss:.4f}")

    return epoch_loss / len(train_dataloader)


def evaluate_model(model, test_dataloader, criterion, device):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_dataloader):
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()

    return test_loss / len(test_dataloader)


def define_dataloaders(dataset_name, batch_size):
    if dataset_name == "dummy":
        train_dataloader, test_dataloader = get_dummy_dataloaders(batch_size=batch_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return train_dataloader, test_dataloader


def main(epochs, learning_rate, batch_size, dataset_name, device):
    model = load_model()
    train_dataloader, test_dataloader = define_dataloaders(dataset_name, batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = CompressedSpectralLoss()

    model.to(device)

    best_loss = float("inf")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_model(model, train_dataloader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}")

        test_loss = evaluate_model(model, test_dataloader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}")

        # Save the model after each epoch
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), f"models/lstm_filter_{dataset_name}.pth")
            print(f"Model saved at epoch {epoch + 1} with loss {best_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the LSTM filter network.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs to train.")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate for training.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training and evaluation.")
    parser.add_argument("--dataset", type=str, default="dummy", help="Dataset to use for training and evaluation.")

    args = parser.parse_args()
    
    if not os.path.exists("models"):
        os.makedirs("models")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train and evaluate the model
    main(args.epochs, args.learning_rate, args.batch_size, args.dataset, device)