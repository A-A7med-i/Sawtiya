import torch
import torch.nn as nn
from tqdm.auto import tqdm
from typing import Callable
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score


class AudioTrainer:
    """
    Trainer class for PyTorch audio models.

    Handles:
        - Training and validation loops
        - Accuracy computation
        - History tracking and plotting
        - Model checkpointing
        - Inference on test data
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        checkpoint_path: str,
    ):
        """
        Initialize the AudioTrainer.

        Args:
            model (nn.Module): PyTorch model to train.
            loss_fn (Callable): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            epochs (int): Number of epochs to train.
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            test_loader (DataLoader): Test data loader.
            checkpoint_path (str): Path to save the best model checkpoint.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.checkpoint_path = checkpoint_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.best_val_acc = float("-inf")

    def compute_accuracy(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> tuple[int, int]:
        """
        Compute number of correct predictions.

        Args:
            logits (torch.Tensor): Model output logits.
            labels (torch.Tensor): True labels.

        Returns:
            Tuple[int, int]: Number of correct predictions, total samples.
        """
        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum().item()
        total = labels.size(0)
        return correct, total

    def run_epoch(self, train: bool, dataloader: DataLoader) -> tuple[float, float]:
        """
        Run one epoch for training or evaluation.

        Args:
            train (bool): True for training, False for evaluation.
            dataloader (DataLoader): DataLoader to iterate over.

        Returns:
            Tuple[float, float]: Average loss and accuracy (%) for the epoch.
        """
        self.model.train() if train else self.model.eval()
        context = torch.enable_grad() if train else torch.inference_mode()

        running_loss, count = 0.0, 0
        total_correct, total_samples = 0, 0
        phase = "Training" if train else "Validation"

        with context:
            for features, labels in tqdm(dataloader, desc=phase):
                features, labels = features.to(self.device), labels.to(self.device)

                logits = self.model(features)
                loss = self.loss_fn(logits, labels)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                correct, total = self.compute_accuracy(logits, labels)
                total_correct += correct
                total_samples += total
                running_loss += loss.item()
                count += 1

        avg_loss = running_loss / count
        accuracy = (total_correct / total_samples) * 100
        return avg_loss, accuracy

    def plot_training_history(self, history: dict):
        """
        Plot training and validation loss and accuracy curves.

        Args:
            history (dict): Dictionary with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        """
        epochs = range(1, len(history["train_loss"]) + 1)
        plt.figure(figsize=(20, 8))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(
            epochs,
            history["train_loss"],
            label="Train Loss",
            marker="o",
            color="#2E86AB",
        )
        plt.plot(
            epochs,
            history["val_loss"],
            label="Validation Loss",
            marker="^",
            linestyle="--",
            color="#E27D60",
        )
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(
            epochs,
            history["train_acc"],
            label="Train Accuracy",
            marker="s",
            color="#28B463",
        )
        plt.plot(
            epochs,
            history["val_acc"],
            label="Validation Accuracy",
            marker="D",
            linestyle="--",
            color="#CA6F1E",
        )
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def train_model(self) -> dict:
        """
        Run full training and validation cycles, saving the best model.

        Returns:
            dict: Training history containing losses and accuracies.
        """
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        for epoch in range(self.epochs):
            print(f"\nEpoch [{epoch + 1}/{self.epochs}]")

            train_loss, train_acc = self.run_epoch(
                train=True, dataloader=self.train_loader
            )
            val_loss, val_acc = self.run_epoch(train=False, dataloader=self.val_loader)

            # Save checkpoint if validation improves
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(
                    f"Checkpoint saved at epoch {epoch + 1} (Validation Accuracy: {val_acc:.2f}%)"
                )

            # Record history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.2f}%")

        return history

    def run_inference(self, dataloader: DataLoader) -> dict:
        """
        Run inference on the given DataLoader.

        Args:
            dataloader (DataLoader): Test or evaluation DataLoader.

        Returns:
            dict: Dictionary with 'accuracy' and 'f1' score.
        """
        self.model.eval()
        all_labels, all_preds, all_probs = [], [], []

        with torch.inference_mode():
            for features, labels in tqdm(dataloader, desc="[Inference]", leave=False):
                features = features.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                logits = self.model(features)
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")

        return {"accuracy": acc, "f1": f1}
