"""Training/evaluation loop for the AG News classifier.

A plain PyTorch loop (no Trainer abstraction) so the mechanics stay visible: each
epoch trains over the train DataLoader, then evaluates accuracy and macro-F1 on
the validation set.
"""

import torch
from sklearn.metrics import accuracy_score, f1_score


class ClassifierTrainer:
    def __init__(self, model, train_loader, val_loader, learning_rate, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def train(self, epochs: int) -> list[dict]:
        history = []
        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch()
            metrics = self.evaluate()
            metrics = {"epoch": epoch, "train_loss": train_loss, **metrics}
            history.append(metrics)
            print(
                f"epoch {epoch}/{epochs} | train_loss {train_loss:.4f} | "
                f"val_acc {metrics['accuracy']:.4f} | val_macro_f1 {metrics['macro_f1']:.4f}"
            )
        return history

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()
            output = self.model(**batch)
            output.loss.backward()
            self.optimizer.step()
            total_loss += output.loss.item()
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        preds, labels = [], []
        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            output = self.model(**batch)
            preds.extend(output.logits.argmax(dim=-1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
        }
