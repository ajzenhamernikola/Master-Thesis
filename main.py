import torch

import src.models.gcn as gcn
import src.models.dgcnn_my as dgcnn


def main():
    # Set the device
    train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_device = torch.device("cpu")
    test_device = torch.device("cpu")

    model_path, log = gcn.train(train_device, test_device)
    gcn.test(train_device, test_device, model_path, log)

    #model_path, log = dgcnn.train(train_device, test_device)
    #dgcnn.test(train_device, test_device, model_path, log)


if __name__ == "__main__":
    main()
