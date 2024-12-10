import torch
from torch import nn
import torchvision

import requests
from pathlib import Path
import random

from helper_functions import accuracy_fn
from helper_functions import print_train_time
from timeit import default_timer as timer

from tqdm.auto import tqdm
import pandas as pd

from pathlib import Path
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


# Import matplotlib for visualization
import matplotlib.pyplot as plt

print(f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")
device = "cuda" if torch.cuda.is_available() else "cpu"
device

####################
# Data preparation
####################
train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor(), target_transform=None)
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

image, label = train_data[0]
class_names = train_data.classes

torch.manual_seed(42)

####################
# Load data for model
####################
BATCH_SIZE = 32

train_dataloader = DataLoader(train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_dataloader = DataLoader(test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)

train_features_batch, train_labels_batch = next(iter(train_dataloader))
print(train_features_batch.shape), print(train_labels_batch.shape)


####################
# Model setup
####################

class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(nn.Flatten(),
                            nn.Linear(in_features=input_shape, out_features=hidden_units),
                            nn.Linear(in_features=hidden_units, out_features=output_shape))
        
    def forward(self, x):
        return self.layer_stack(x)
    
model_0 = FashionMNISTModelV0(input_shape=784,
    hidden_units=10,
    output_shape=len(class_names)
)
model_0.to("cpu")

#--------------------------------

class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(nn.Flatten(),
                    nn.Flatten(),
                    nn.Linear(in_features=input_shape, out_features=hidden_units),
                    nn.ReLU(),
                    nn.Linear(in_features=hidden_units, out_features=output_shape),
                    nn.ReLU()
                    )
        
    def forward(self, x):
        return self.layer_stack(x)
    
model_1 = FashionMNISTModelV1(input_shape=784,
    hidden_units=10,
    output_shape=len(class_names)
)
model_1.to("cpu")

#-------------------------------

class FashionMNISTModelV2(nn.Module):
    """
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our input data.
            nn.Linear(in_features=hidden_units*7*7, 
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x
    
model_2 = FashionMNISTModelV2(input_shape=1, 
    hidden_units=10, 
    output_shape=len(class_names)).to(device)
model_2.to("cpu")

loss_fn = nn.CrossEntropyLoss()
optimizer0 = torch.optim.SGD(params=model_0.parameters(), lr=0.1)
optimizer1 = torch.optim.SGD(params=model_1.parameters(), lr=0.1)
optimizer2 = torch.optim.SGD(params=model_2.parameters(), lr=0.1)


####################
# Training model
####################
torch.manual_seed(42)
train_time_start_on_cpu = timer()
epochs = 3

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1))

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode(): 
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            
            test_pred = model(X)
            
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred.argmax(dim=1)
            )
        
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

torch.manual_seed(42)
def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn, 
               device: torch.device = device):

    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model_name": model.__class__.__name__,
            "model_loss": loss.item(),
            "model_acc": acc}

# for epoch in tqdm(range(epochs)):
#     print(f"Epoch: {epoch}\n---------")
#     train_step(data_loader=train_dataloader, 
#         model=model_0, 
#         loss_fn=loss_fn,
#         optimizer=optimizer0,
#         accuracy_fn=accuracy_fn
#     )
#     test_step(data_loader=test_dataloader,
#         model=model_0,
#         loss_fn=loss_fn,
#         accuracy_fn=accuracy_fn
#     )

# total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu, 
#                                            end=timer(),
#                                            device="cpu")
# model_0_results = eval_model(model=model_0, data_loader=test_dataloader,
#     loss_fn=loss_fn, accuracy_fn=accuracy_fn,
#     device=device
# )
# train_time_start_on_cpu = timer()

# for epoch in tqdm(range(epochs)):
#     print(f"Epoch: {epoch}\n---------")
#     train_step(data_loader=train_dataloader, 
#         model=model_1, 
#         loss_fn=loss_fn,
#         optimizer=optimizer1,
#         accuracy_fn=accuracy_fn
#     )
#     test_step(data_loader=test_dataloader,
#         model=model_1,
#         loss_fn=loss_fn,
#         accuracy_fn=accuracy_fn
#     )

# total_train_time_model_1 = print_train_time(start=train_time_start_on_cpu, 
#                                            end=timer(),
#                                            device="cpu")
# model_1_results = eval_model(model=model_1, data_loader=test_dataloader,
#     loss_fn=loss_fn, accuracy_fn=accuracy_fn,
#     device=device
# )
# train_time_start_on_cpu = timer()

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=train_dataloader, 
        model=model_2, 
        loss_fn=loss_fn,
        optimizer=optimizer2,
        accuracy_fn=accuracy_fn
    )
    test_step(data_loader=test_dataloader,
        model=model_2,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn
    )

total_train_time_model_2 = print_train_time(start=train_time_start_on_cpu, 
                                           end=timer(),
                                           device=str(next(model_2.parameters()).device))
model_2_results = eval_model(model=model_2, data_loader=test_dataloader,
    loss_fn=loss_fn, accuracy_fn=accuracy_fn,
    device=device
)

# compare_results = pd.DataFrame([model_0_results, model_1_results, model_2_results])
# print(compare_results)

# compare_results["training_time"] = [total_train_time_model_0, total_train_time_model_1, total_train_time_model_2]
# print(compare_results)

# compare_results.set_index("model_name")["model_acc"].plot(kind="barh")
# plt.xlabel("accuracy (%)")
# plt.ylabel("model")
# plt.show()

test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)


def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device)

            pred_logit = model(sample)

            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

            pred_probs.append(pred_prob.cpu())
            
    return torch.stack(pred_probs)

pred_probs = make_predictions(model = model_2, data = test_samples)
print(pred_probs[:2])

pred_classes = pred_probs.argmax(dim=1)
print(pred_classes)

plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):
  plt.subplot(nrows, ncols, i+1)

  plt.imshow(sample.squeeze(), cmap="gray")

  pred_label = class_names[pred_classes[i]]

  truth_label = class_names[test_labels[i]] 

  title_text = f"Pred: {pred_label} | Truth: {truth_label}"
  
  if pred_label == truth_label:
      plt.title(title_text, fontsize=10, c="g")
  else:
      plt.title(title_text, fontsize=10, c="r")
  plt.axis(False)
plt.show()

MODEL_PATH = Path("model")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "fashion_computer_vision.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_2.state_dict(), f=MODEL_SAVE_PATH)