import torch
from exp_cifar.cifar_dataset import cifar10_loader
from nn.nn_utils import load_model
from models.cifar_tiny import Cifar_Tiny
from models.resnet import ResNet18
from time import time
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(name, conf_matrix, class_names=range(1,10)):
    """
    Plot the confusion matrix as a heatmap.
    
    :param conf_matrix: Tensor or ndarray, the confusion matrix.
    :param class_names: List of class names for axis labels.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix " + name)
    plt.savefig("figs/CM_" + name + ".png")

pathes = ["models/resnet18_cifar10.model",\
    "models/tiny_cifar10.model",\
    "models/cifar_tiny_resnet18_cifar10_distill_cifar10.model",\
    "models/cifar_tiny_resnet18_cifar10_hint_optimized_cifar10.model",\
    "models/cifar_tiny_resnet18_cifar10_kt_cifar10.model"]
names = ["Teacher",\
    "Student",\
    "KD",\
    "FitNet",\
    "KT"]

transfer_loader = cifar10_loader(batch_size=128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loader = transfer_loader[1]

for path, name in zip(pathes, names):
    print(f"Algorithm: ", name)
    if path == "models/resnet18_cifar10.model":
        net = ResNet18(10)
    else:
        net = Cifar_Tiny(10)
    state_dict = torch.load(path, weights_only = False)
    net.load_state_dict(state_dict)
    net.eval()
    net.to(device)
    # Measure inference time
    start_time = time()
    torch.cuda.reset_peak_memory_stats(device) 
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        output = net(inputs)
    end_time = time()
    memory_usage = torch.cuda.max_memory_allocated(device)
    print(f"Memory usage: {memory_usage/1024/1024:.2f} MB")
    inference_time = end_time - start_time
    print(f"Inference Time: {inference_time:.6f} seconds")

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    total = 0
    correct = 0
    conf_matrix = np.zeros((10, 10), dtype=np.int64)
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)  # Get predicted class
        for true_label, pred_label in zip(labels, predicted):
            conf_matrix[true_label, pred_label] += 1
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    plot_confusion_matrix(name, conf_matrix)

    TP = np.diag(conf_matrix)

    FP = np.sum(conf_matrix, axis=0) - TP

    FN = np.sum(conf_matrix, axis=1) - TP

    precision = np.sum(TP / (TP + FP + 1e-8))/10*100
    print(f"Macro Average Precision: {precision:.2f}%")
