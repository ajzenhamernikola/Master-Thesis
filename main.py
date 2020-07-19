import torch

import src.models.gcn as gcn


if __name__ == "__main__":
    # Set the device
    train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # train_device = torch.device("cpu")
    test_device = torch.device("cpu")

    model_path, log = gcn.train(train_device, test_device)
    gcn.test(train_device, test_device, model_path, log)
