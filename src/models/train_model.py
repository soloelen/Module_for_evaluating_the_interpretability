from copy import deepcopy

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from ..config.config import DEVICE, NUM_EPOCHS


def train(model,
          train_loader,
          valid_dataloader,
          optimizer,
          criterion,
          lr_scheduler=None):
    """Обучение модели."""

    model = model.to(DEVICE)
    best_val_acc = -1000
    best_model_wts = deepcopy(model.state_dict())
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(NUM_EPOCHS):

        model.train(True)
        running_loss = 0.0
        running_acc = 0
        num_train = 0

        # train_iterator = tqdm(enumerate(train_loader))
        with tqdm(total=len(train_loader)) as progress_bar:
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                out = torch.argmax(outputs.detach(), dim=1)
                assert out.shape == labels.shape
                running_acc += (labels == out).sum().item()
                num_train += inputs.size(0)
                progress_bar.update(1)

        train_loss = running_loss / num_train
        train_acc = running_acc * 100 / num_train
        print(f"Epoch {epoch + 1}")
        print(f"Train loss: {train_loss}, Train Acc: {train_acc}%")

        running_val_loss = 0.0
        correct = 0
        num_valid = 0
        model.eval()
        with torch.no_grad():
            with tqdm(total=len(valid_dataloader)) as progress_bar:
                for inputs, labels in valid_dataloader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item() * inputs.size(0)
                    outputs = torch.argmax(outputs, dim=1)
                    acc = (outputs == labels).sum().item()
                    correct += acc
                    num_valid += inputs.size(0)
                    progress_bar.update(1)

        val_loss = running_val_loss / num_valid
        val_acc = correct * 100 / num_valid
        print(f"Val loss: {val_loss}, Val Acc: {val_acc}%")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if correct > best_val_acc:
            best_val_acc = correct
            best_model_wts = deepcopy(model.state_dict())
        if lr_scheduler is not None:
            lr_scheduler.step()

    model.load_state_dict(best_model_wts)
    print('Finished Training')
    return history


def show_metrics(history, title):
    """
    Визуализация метрик после обучения.
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(history['train_loss'])
    axs[0].plot(history['val_loss'])
    axs[0].title.set_text('Training Loss vs Validation Loss')
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend(['Training', 'Validation'])
    axs[1].plot(history['train_acc'])
    axs[1].plot(history['val_acc'])
    axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
    axs[1].legend(['Training', 'Validation'])
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    fig.suptitle(title)
    plt.show()
