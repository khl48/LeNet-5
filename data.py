from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageOps

import torch
import numpy as np
import pandas as pd

import io
import pathlib
import subprocess
import shutil
import random
import zipfile

MNIST_TRAIN  = "./student_data/mnist_train.pt"
MNIST_TEST   = "./student_data/mnist_test.pt"
MNIST_TRAIN2 = "./student_data/mnist_train2.pt"

DIGITS_DATA = "./student_data/digits.pt"

class HuggingFaceMNIST(Dataset):
    def __init__(self, train=True):
        train_url = "hf://datasets/ylecun/mnist/mnist/train-00000-of-00001.parquet"
        test_url = "hf://datasets/ylecun/mnist/mnist/test-00000-of-00001.parquet"
        
        if not train:
            df = pd.read_parquet(test_url)
        else:
            df = pd.read_parquet(train_url)

        preprocess = transforms.Compose([
            transforms.Pad((2, 2), padding_mode="constant"),
            transforms.ToTensor(),
        ])

        x = []
        y = []

        for _, row in df.iterrows():
            img = row["image"]["bytes"]
            label = row["label"]
            buffer = preprocess(Image.open(io.BytesIO(img)))

            x.append(buffer)
            y.append(label)

        self.data = { "x": x, "y": y }

    def __len__(self):
        return len(self.data["x"])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (self.data["x"][idx], self.data["y"][idx])

class HuggingFaceMNIST2(Dataset):
    def __init__(self, train=True):
        train_url = "hf://datasets/ylecun/mnist/mnist/train-00000-of-00001.parquet"
        test_url = "hf://datasets/ylecun/mnist/mnist/test-00000-of-00001.parquet"
        
        if not train:
            df = pd.read_parquet(test_url)
        else:
            df = pd.read_parquet(train_url)

        preprocess = transforms.Compose([
            transforms.Pad((2, 2), padding_mode="constant"),
            transforms.ToTensor(),
        ])

        aug_preprocess = transforms.Compose([
            transforms.RandomAffine(
                degrees=(-30,30),
                translate=(0.2, 0.2),
                scale=(0.8, 1.2),
                shear=(-15, 15),
            ),
            transforms.Pad((2, 2), padding_mode="constant"),
            transforms.ToTensor(),
        ])

        x = []
        y = []
        
        import matplotlib.pyplot as plt
        for _, row in df.iterrows():
            img = row["image"]["bytes"]
            label = row["label"]
            buffer = preprocess(Image.open(io.BytesIO(img)))

            if random.random() >= 0.6:
                aug_buffer = aug_preprocess(Image.open(io.BytesIO(img)))
                x.append(aug_buffer)

                #image = np.array(aug_buffer).reshape(32, 32)
                #plt.imshow(image, cmap="gray")
                #plt.show()

                y.append(label)

            x.append(buffer)
            y.append(label)

        self.data = { "x": x, "y": y }

    def __len__(self):
        return len(self.data["x"])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (self.data["x"][idx], self.data["y"][idx])

class KaggleDigits:
    def __init__(self):
        data_path = pathlib.Path(DIGITS_DATA)
        data_dir = data_path.parent
        
        work_dir = data_dir / "digits"
        zip_path = work_dir / "digits.zip"
        
        if work_dir.exists() and work_dir.is_dir():
            shutil.rmtree(work_dir)

        work_dir.mkdir(parents=True, exist_ok=True)

        subprocess.run([
            "curl",
            "-L",
            "-o",
            zip_path,
            "https://www.kaggle.com/api/v1/datasets/download/karnikakapoor/digits",
        ])

        with zipfile.ZipFile(zip_path) as zip_ref:
            zip_ref.extractall(work_dir)

        zip_path.unlink(missing_ok=True)

        png_dir = work_dir / "digits updated"
        jpeg_dir = work_dir / "digits_jpeg"

        for digit in (png_dir / "digits updated").iterdir():
            shutil.move(digit, work_dir)

        if png_dir.exists() and png_dir.is_dir():
            shutil.rmtree(png_dir)

        if jpeg_dir.exists() and jpeg_dir.is_dir():
            shutil.rmtree(jpeg_dir)

        preprocess = transforms.Compose([
            # Note: PIL inverts before this preprocess
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((12, 7)),
            transforms.ToTensor(),
        ])

        by_class = [[] for _ in range(10)]
        for digit in work_dir.iterdir():
            try:
                idx = int(digit.name)
            except ValueError:
                continue

            if not (0 <= idx <= 9):
                continue

            for img in digit.iterdir():
                if img.name.endswith(".png"):
                    img = ImageOps.invert(Image.open(img))
                    buffer = preprocess(img)
                    by_class[idx].append(np.array(buffer))

        if work_dir.exists() and work_dir.is_dir():
            shutil.rmtree(work_dir)

        self.representatives = [torch.Tensor() for _ in range(10)]
        for idx in range(len(by_class)):
            shape = None
            data = by_class[idx]
            for j in range(len(data)):
                img = data[j][0]
                shape = img.shape
                data[j] = img.flatten()

            rep = np.mean(data, axis=0) * 1.5
            np.clip(rep[..., 0], 0, 255, out=rep[...,0])

            self.representatives[idx] = torch.from_numpy(rep)

            for j in range(len(by_class)):
                img = data[j]
                data[j] = img.reshape(shape)

    def __getitem__(self, idx):
        return self.representatives[idx]

if __name__ == "__main__":
    mnist_train_path  = pathlib.Path(MNIST_TRAIN)
    mnist_test_path   = pathlib.Path(MNIST_TEST)
    mnist_train2_path = pathlib.Path(MNIST_TRAIN2)
    digits_path = pathlib.Path(DIGITS_DATA)

    prompt = input("Generate dataset files? (y/[n]) ")
    if prompt.lower() != "y":
        exit()
    else:
        mnist_train_path.unlink(missing_ok=True)
        mnist_test_path.unlink(missing_ok=True)
        mnist_train2_path.unlink(missing_ok=True)
        digits_path.unlink(missing_ok=True)

    mnist_train  = HuggingFaceMNIST(train=True)
    mnist_test   = HuggingFaceMNIST(train=False)
    mnist_train2 = HuggingFaceMNIST2(train=True)

    mnist_train_path.parent.mkdir(parents=True, exist_ok=True)
    mnist_test_path.parent.mkdir(parents=True, exist_ok=True)
    mnist_train2_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mnist_train, mnist_train_path)
    torch.save(mnist_test, mnist_test_path)
    torch.save(mnist_train2, mnist_train2_path)

    print("Saved MNIST dataset to", str(mnist_train_path), "and", str(mnist_test_path))

    digits = KaggleDigits()

    digits_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(digits, digits_path)

    print("Saved Digits to", digits_path)
