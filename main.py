import os
import torch

import constants
from data.StartingDataset import StartingDataset
from networks.StartingNetwork import StartingNetwork
from train_functions.starting_train import starting_train

# CURRENTLY——we are indeed getting KeyErrors. such as "Bitsat", not in the glove vocabulary.
# Birla Institute of Technology & Science Admission Test (BITSAT)

#

def main():
    # Get command line arguments
    hyperparameters = {"epochs": constants.EPOCHS, "batch_size": constants.BATCH_SIZE}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Epochs:", constants.EPOCHS)
    print("Batch size:", constants.BATCH_SIZE)

    # Initalize dataset and model. Then train the model!
    data_path = "/Users/Terru/Desktop/UCLA/ACM AI/Projects/train.csv"

    # TO-DO: train, test, (val) split ofc

    train_dataset = StartingDataset(data_path)
    val_dataset = StartingDataset(data_path)
    model = StartingNetwork(300, 1024, 400000, 2, 3)
    # hyperparameters more or less arbitrary. At least they are for now, I just randomly set them
    # vocab size of glove.6B that we're using is 400K

    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=constants.N_EVAL,
        device=device,
    )

if __name__ == "__main__":
    main()