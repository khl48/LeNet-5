from torch import nn
from torch.nn import init
from torch.utils.data import DataLoader
from data import *

import torch
import numpy as np

MNIST_TRAIN = "./student_data/mnist_train2.pt"
MNIST_TEST  = "./student_data/mnist_test.pt"

DIGITS_DATA = "./student_data/digits.pt"

MODEL_SAVEFILE = "./LeNet5_2.pth"

EPOCHS = 20
BATCH_SIZE = 1
LR = 1e-3
LOSS_REGULARIZATION = 0.1 # j in equation (9) loss function

class Subsampling(nn.Module):
    def __init__(self, input_channels, kernel_size, stride):
        super(Subsampling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.coef = nn.Parameter(torch.ones(input_channels))
        self.bias = nn.Parameter(torch.zeros(input_channels))

    def forward(self, x):
        avg_pool = nn.functional.avg_pool2d(x, self.kernel_size, self.stride)
        scaled = avg_pool * self.coef.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

        return scaled

class LeNet5Loss(nn.Module):
    def __init__(self):
        super(LeNet5Loss, self).__init__()

    def forward(self, logits, targets):
        p = logits.size(0)

        correct_class_scores = logits[range(p), targets]
        
        bias = torch.tensor(-LOSS_REGULARIZATION).exp().to(logits.device)
        logits = torch.cat([logits, bias.expand(p, 1)], dim=1)

        log_sum_exp = torch.logsumexp(logits, dim=1)
        
        return (log_sum_exp - correct_class_scores).mean()

class RBFOutputLayer(nn.Module):
    def __init__(self, input_size, num_classes, representatives):
        super(RBFOutputLayer, self).__init__()
        self.representatives = representatives
        self.input_size = input_size
        self.num_classes = num_classes
    
    def forward(self, x):
        total_distances = []

        for i in range(self.num_classes):
            rep = self.representatives[i]
            if rep.size(0) != self.input_size:
                raise ValueError("Incorrect RBF input")
            
            dist = torch.sum((x - rep) ** 2, dim=1)
            total_distances.append(dist)

        return torch.stack(total_distances, dim=1)

class TanHyperbolic(nn.Module):
    def __init__(self):
        super(TanHyperbolic, self).__init__()

    def forward(self, x):
        return 1.7159 * torch.tanh(x * 2/3)

class LeNet5(nn.Module):
    def __init__(self, representatives):
        super(LeNet5, self).__init__()

        self.c1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0) # 28x28
        self.s2 = Subsampling(6, kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.s4 = Subsampling(16, kernel_size=2, stride=2)
        self.c5 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        self.f6 = nn.Linear(120, 84)
        self.out = RBFOutputLayer(84, 10, representatives)

        # https://stackoverflow.com/questions/42670274/how-to-calculate-fan-in-and-fan-out-in-xavier-initialization-for-neural-networks
        LeNet5.uniform_init_param(self.c1.parameters, a=-2.4/25, b=2.4/25)
        LeNet5.uniform_init_param(self.s2.parameters, a=-2.4/24, b=2.4/24)
        LeNet5.uniform_init_param(self.c3.parameters, a=-2.4/150, b=2.4/150)
        LeNet5.uniform_init_param(self.s4.parameters, a=-2.4/64, b=2.4/64)
        LeNet5.uniform_init_param(self.c5.parameters, a=-2.4/400, b=2.4/400)
        LeNet5.uniform_init_param(self.f6.parameters, a=-2.4/120, b=2.4/120)

        self.model = nn.Sequential(
            # Section 2B
            # Layer C1, activation comes from equation 6
            self.c1,
            nn.ReLU(),
            # Layer S2
            # pg. 6, local average and sub-sampling with trainable parameters
            self.s2,
            nn.ReLU(),
            # Layer C3
            self.c3,
            nn.ReLU(),
            # Layer S4
            self.s4,
            nn.ReLU(),
            # Layer C5
            self.c5,
            nn.ReLU(),
            nn.Flatten(),
            # Layer F6
            nn.Dropout(0.5),
            self.f6,
            nn.ReLU(),
            # Output RBF Layer
            self.out,
        )
    
    @staticmethod
    def uniform_init_param(params_fn, a, b):
        for param in params_fn():
            if param.dim() > 1:
                init.uniform_(param, a=a, b=b)

    def forward(self, x):
        return self.model(x)

def test_during_test(dataloader,model):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_accuracy = 0
    length = 0
    for _, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        if torch.argmax(pred, dim=1) == y:
            test_accuracy += 1
        length += 1
    test_accuracy /= length
    model.train()
    return test_accuracy

def main():
    mnist_train : HuggingFaceMNIST = torch.load(MNIST_TRAIN, weights_only=False)
    mnist_test  : HuggingFaceMNIST = torch.load(MNIST_TEST, weights_only=False)
    digits      : KaggleDigits     = torch.load(DIGITS_DATA, weights_only=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(len(digits.representatives)):
        digits.representatives[i] = digits.representatives[i].to(device)

    model = LeNet5(digits).to(device)

    print("Using", device)

    mnist_train_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    epoch_errors = []
    n_correct = 0
    n_total = 0
    for i in range(EPOCHS):
        print("Computing epoch", i + 1)
        for i, (x, y) in enumerate(mnist_train_loader):
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pred_labels = torch.argmax(pred, dim=1)
            n_correct += (pred_labels == y).sum().item()

            n_total += 1

            if i % 5000 == 0:
                loss = loss.item()
                print(f"- loss: {loss:>7f} at {i + 1}")
        epoch_errors.append((n_correct, n_total, n_correct / n_total))
        mnist_test_loader = DataLoader(mnist_test)

        model.eval()
        print("- error rate for epoch:", test_during_test(mnist_test_loader, model))
        model.train()
    
    print("Epoch    | (n_correct, n_total, accuracy)")
    for i in range(len(epoch_errors)):
        print("Epoch", i+1, ":", epoch_errors[i])

    model.eval()
    torch.save(model, MODEL_SAVEFILE)

if __name__ == "__main__":
    main()
