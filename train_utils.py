import torch
import torch.nn as nn
from time import time
from sklearn.metrics import accuracy_score
import copy

def train_classifier(model, train_loader, valid_loader, optimizer, criterion, device, num_epochs=10, logging_interval=50, patience=3):
    history = {
        'train_loss': [],
        'valid_loss': [],
        'train_acc': [],
        'valid_acc': []
    }

    for epoch in range(1, num_epochs + 1):
        start_time = time()

        # Training
        model.train()
        train_loss, train_preds, train_targets = 0.0, [], []

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

            if (batch_idx + 1) % logging_interval == 0:
                print(f"Epoch {epoch:03d} | Batch {batch_idx+1:04d}/{len(train_loader):04d} | Loss: {loss.item():.4f}")

        train_acc = accuracy_score(train_targets, train_preds)
        avg_train_loss = train_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        valid_loss, valid_preds, valid_targets = 0.0, [], []

        with torch.no_grad():
            for imgs, labels in valid_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                valid_loss += loss.item() * imgs.size(0)
                valid_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                valid_targets.extend(labels.cpu().numpy())

        valid_acc = accuracy_score(valid_targets, valid_preds)
        avg_valid_loss = valid_loss / len(valid_loader.dataset)

        history['train_loss'].append(avg_train_loss)
        history['valid_loss'].append(avg_valid_loss)
        history['train_acc'].append(train_acc)
        history['valid_acc'].append(valid_acc)

        elapsed = (time() - start_time) / 60
        print(f"***Epoch {epoch:03d} | Train Acc: {train_acc*100:.2f}% | Loss: {avg_train_loss:.3f}")
        print(f"***Epoch {epoch:03d} | Valid Acc: {valid_acc*100:.2f}% | Loss: {avg_valid_loss:.3f}")
        print(f"Time elapsed: {elapsed:.2f} min\n")
    return history