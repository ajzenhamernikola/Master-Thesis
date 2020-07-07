import torch

import src.models.gcn as gcn


if __name__ == "__main__":
    # Set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_device = torch.device("cpu")

    model_path = gcn.train(device)
    gcn.test(device, test_device, model_path)