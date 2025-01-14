import torch
import matplotlib.pyplot as plt
import numpy as np
from Data_Collection.data_normalization import unnormalize_tensor
from upsampling import joint_bilateral_up_sample

def single_pass(model, input_tensor, guide_tensor, device, dataset_mean=0, dataset_std=0, target_tensor = None, display_plot=False, display_target=False, print_tensor=False, display_sample=False, save_plot=False, plot_dir=""):

    model = model.to(device)
    input_tensor = input_tensor.to(device)

    result = model(input_tensor)

    result = result.detach()
    result = result.to(torch.device("cpu"))
    result = unnormalize_tensor(result, dataset_mean, dataset_std)

    result = result.detach().numpy()
    result = result.squeeze(0)

    guide_tensor = guide_tensor.detach().numpy()

    if guide_tensor.shape[0] == 4:
        guide_tensor = guide_tensor[:3, :, :]

    result = joint_bilateral_up_sample(result, guide_tensor, save_img=save_plot, output_path=plot_dir)

    if display_plot:

        plt.imshow(result)
        plt.title("Generated Map")
        plt.show()

    if display_sample:
        input_tensor = input_tensor.squeeze(0)
        input_tensor = input_tensor.to(torch.device("cpu"))
        img = input_tensor.detach()
        img = np.transpose(img, (1,2,0))
        plt.imshow(img)
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

    if print_tensor:
        print(result.shape)
        print(result)