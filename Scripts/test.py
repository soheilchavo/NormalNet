import torch
import matplotlib.pyplot as plt
import numpy
from Data_Collection.data_normalization import unnormalize_tensor

def single_pass(model, input_tensor, device, dataset_mean=0, dataset_std=0, display_plot=False, print_tensor=False, display_sample=False):

    model = model.to(device)
    input_tensor = input_tensor.to(device)

    result = model(input_tensor)

    result = result.squeeze(0)

    if print_tensor:
        print(result.shape)
        print(result)

    if display_sample:
        input_tensor = input_tensor.squeeze(0)
        img = input_tensor.detach()
        img = unnormalize_tensor(img, dataset_mean, dataset_std)
        img = img.clip(0, 1)
        plt.imshow(img.permute(1, 2, 0))
        plt.title("Input Image")
        plt.show()

    if display_plot:
        img = result.detach()
        img = unnormalize_tensor(img, dataset_mean, dataset_std)
        img = img.clip(0, 1)
        plt.imshow(img.permute(1, 2, 0))
        plt.title("Generated Map")
        plt.show()