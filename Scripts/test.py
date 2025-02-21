import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
from Data_Collection.data_normalization import unnormalize_tensor, scale_transform_sample
from Data_Collection.data_filtering import transform_single_png
from Scripts.upsampling import joint_bilateral_up_sample

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

    if result[0].mean() > 1:
        result /= 255

    result = joint_bilateral_up_sample(result, guide_tensor, save_img=save_plot, output_path=plot_dir)

    if display_plot:
        plt.imshow(result)
        plt.title("Generated Map")
        plt.show()

    if display_sample:
        img = np.transpose(guide_tensor, (1,2,0))
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

def generate_pbr(model_strings, input_tensor, guide_tensor, device, display_plots = True, save_plots=False, save_dir=""):

    for i in model_strings:

        with open(f'Data/{i}TrainingDatasetInfo', 'rb') as f:
            values = pickle.load(f)

        dataset_mean = values[0]
        dataset_std = values[1]

        generator = torch.load(f"Models/{i}Generator.pt")

        single_pass(generator, input_tensor, guide_tensor, device, dataset_mean, dataset_std, save_plot=save_plots, plot_dir=save_dir+i+'.png', display_plot=display_plots)

def test_single_sample(sample_dir, gen_type, generator, device, display_plot=False, display_sample=False, save_plot=True, plot_dir="", print_tensor=False):

    sample = transform_single_png(sample_dir)

    down_sample = scale_transform_sample(sample, standalone=True)

    with open(f'Data/{gen_type}TrainingDatasetInfo', 'rb') as f:
        values = pickle.load(f)

    dataset_mean = values[0]
    dataset_std = values[1]

    single_pass(model=generator, input_tensor=down_sample, guide_tensor=sample, device=device, dataset_mean=dataset_mean, dataset_std=dataset_std, display_plot=display_plot, display_sample=display_sample, save_plot=save_plot, plot_dir=plot_dir, print_tensor=print_tensor)

def load_gan(gen_path, disc_path, device):
    generator = torch.load(gen_path, map_location=device)
    discriminator = torch.load(disc_path, map_location=device)
    return generator, discriminator