import torch
import matplotlib.pyplot as plt
import numpy
from Data_Collection.data_normalization import unnormalize_tensor
from torchvision.utils import save_image

def single_pass(model, input_tensor, device, dataset_mean=0, dataset_std=0, target_tensor = None, display_plot=False, display_target=False, print_tensor=False, display_sample=False, save_plot=False, plot_dir=""):

    model = model.to(device)
    input_tensor = input_tensor.to(device)

    result = model(input_tensor)

    result = result.squeeze(0)

    if print_tensor:
        print(result.shape)
        print(result)

    if save_plot:
        img = result.detach()
        img = img.to(torch.device("cpu"))
        img = unnormalize_tensor(img, dataset_mean, dataset_std)
        img = img.clip(0, 1)
        save_image(img, plot_dir)

    if display_sample:
        input_tensor = input_tensor.squeeze(0)
        input_tensor = input_tensor.to(torch.device("cpu"))
        img = input_tensor.detach()
        img = unnormalize_tensor(img, dataset_mean, dataset_std)
        img = img.clip(0, 1)
        plt.imshow(img.permute(1, 2, 0))
        plt.title("Input Image")
        plt.show()

    if display_target:
        target_tensor = target_tensor.squeeze(0)
        target_tensor = target_tensor.to(torch.device("cpu"))
        img = target_tensor.detach()
        img = unnormalize_tensor(img, dataset_mean, dataset_std)
        img = img.clip(0, 1)
        plt.imshow(img.permute(1, 2, 0))
        plt.title("Target Image")
        plt.show()

    if display_plot:
        img = result.detach()
        img = img.to(torch.device("cpu"))
        img = unnormalize_tensor(img, dataset_mean, dataset_std)
        img = img.clip(0, 1)
        plt.imshow(img.permute(1, 2, 0))
        plt.title("Generated Map")
        plt.show()