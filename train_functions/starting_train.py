import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval, device):
    """
    Trains and evaluates a model.
    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.AdamW(model.parameters())
    loss_fn = nn.BCELoss()

    # change model parameters to float
    model = model.float()

    step = 0
    writer = SummaryWriter() # tensorboard
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for batch in tqdm(train_loader):

            texts, labels = batch

            # Move to GPU if available
            texts = texts.to(device)
            labels = labels.to(device)

            # TODO: Forward propagate
            outputs = model(texts.float())

            # TODO: Backpropagation and gradient descent
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0 and step > 0:
                # TODO:
                writer.add_scalar("Training loss: ", loss.item(), epoch+1)
                # Compute training loss and accuracy.
                # Log the results to Tensorboard.

                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard.
                # Don't forget to turn off gradient calculations!
                writer.add_scalar("Validation accuracy: ", evaluate(val_loader, model, loss_fn, device), epoch+1)
                # NOT DONE. ^ depending on what our evaluate() function actually returns we may need to subset it
                # such as [0]
                evaluate(val_loader, model, loss_fn, device) # testing it here

            writer.flush() # sends output
            step += 1

        print("Epoch: ", epoch+1, "Loss: ", loss.item()) # displays loss of last batch for every epoch
    writer.close()


# def compute_accuracy(outputs, labels):
#     """
#     Computes the accuracy of a model's predictions.
#     Example input:
#         outputs: [0.7, 0.9, 0.3, 0.2]
#         labels:  [1, 1, 0, 1]
#     Example output:
#         0.75
#     """
#
#     n_correct = (torch.round(outputs) == labels).sum().item()
#     n_total = len(outputs)
#     return n_correct / n_total


def evaluate(val_loader, model, loss_fn, device):
    """
    Computes the loss and accuracy of a model on the validation dataset.
    TODO!
    """
    model.eval()

    model = model.to(device)

    correct = 0
    total = 0
    for batch in val_loader:
        texts, labels = batch

        # pass to GPU if available
        texts = texts.to(device)
        labels = labels.to(device)

        # classical accuracy:
        predictions = model(texts).argmax(axis=1)
        correct += (predictions == labels).sum().item()
        total += len(labels)

    print("\n Accuracy: ", 100*(correct/total), "%")

    # TO-DO: calculate ROC accuracy. or separate accuracies for sincere and insincere.
    # sk_learn roc_curve
    # further: try all types like F-1 score and look for anomalies/things to note
    pass