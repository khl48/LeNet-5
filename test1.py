from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import torch
import numpy as np
import torchvision

from data import HuggingFaceMNIST
from train1 import *

import pathlib

def test(dataloader,model):
    #please implement your test code#
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_accuracy = 0
    length = 0
    truth_y = []
    pred_y = []

    fails = [[-500000, None, None] for _ in range(10)]

    greatest_x_fail = None
    greatest_y_fail = -5000
    greatest_pred_fail = None

    for _, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        truth_y.append(y.cpu())
        pred_y.append(torch.argmax(pred, dim=1).cpu())

        if torch.argmax(pred, dim=1) == y:
            test_accuracy += 1
        else:
            fail = np.array(pred.detach().cpu())[0][torch.argmax(pred,dim=1).cpu()]
            if fails[y][0] <= fail:
                fails[y][0] = fail
                fails[y][1] = x.cpu()
                fails[y][2] = torch.argmax(pred,dim=1).item()
        length += 1
    test_accuracy /= length

    import matplotlib.pyplot as plt
    i = 0
    for fail in fails:
        img = np.array(fail[1]).reshape(32, 32)
        print("Real value was", i)
        i += 1
        print("This was predicted as", fail[2])
        plt.imshow(img, cmap="gray")
        plt.show()
    ###########################                                                                                                                                                                               
    
    print(confusion_matrix(truth_y, pred_y))
    print("test accuracy:", test_accuracy)

def main():
    path = pathlib.Path("./student_data/mnist_test.pt")
    if not path.exists():
        print("Please run `python data.py` first to generate ./student_data/")
        print("This implementation detail was made because the MNIST")
        print("referenced in the announcements clarification did not seem to")
        print("work. This assignment instead uses the HuggingFace test")
        print("split that was specified in the initial homework 4 spec.")
        print("\nThis program will now exit.")
        exit(1)
    
    mnist_test : HuggingFaceMNIST = torch.load(path, weights_only=False)

    #pad=torchvision.transforms.Pad(2,fill=0,padding_mode='constant')
    #mnist_test=mnist.MNIST(split="test",transform=pad)
    test_dataloader= DataLoader(mnist_test,batch_size=1,shuffle=False)
    model = torch.load("LeNet5_1.pth", weights_only=False)
    test(test_dataloader,model)
 
if __name__=="__main__":
    main()
