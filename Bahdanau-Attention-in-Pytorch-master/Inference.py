import torch
import pandas as pd


def get_inference(train_mean, train_std, nn_model, x):
    # Load the entire model
    rnn = torch.load(nn_model)

    # Set the model to evaluation mode
    rnn.eval()

    with torch.no_grad():  # Disable gradient computation for inference
        inference = rnn(x)
        inference = (inference - train_mean) / train_std
    return inference

