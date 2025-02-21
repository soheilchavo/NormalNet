#This Project was Created using Photogrammetry PBR's from ambientCG.com,
#licensed under the Creative Commons CC0 1.0 Universal License.

import torch.optim
from Scripts.Data_Collection.data_collector import get_dataset, load_paired_data
from Scripts.Data_Collection.data_filtering import extract_dataset
from Scripts.test import test_single_sample, load_gan
from Scripts.train import train_gan, train_models
import os

#URL's and output directories for getting training and testing data (CSV of all materials and download links)
training_data_info_url = "https://ambientCG.com/api/v2/downloads_csv?method=PBRPhotogrammetry&type=Material&sort=Popular"
testing_data_info_url = ""
training_data_info_output = "Data/training_data_info"
testing_data_info_output = "Data/testing_data_info"
training_raw_data_path = "Data/TrainingRawData"
testing_raw_data_path = "Data/TestingRawData"
training_data_path = "Data/TrainingImages"
testing_data_path = "Data/TestingImages"

#Variables for filtering dataset
data_filter = ["1K-PNG"]
data_heading = "downloadAttribute"
data_types = ["AmbientOcclusion", "Color", "NormalDX", "NormalGL", "Roughness"]

#Parameters
num_data_points = 500
dataset_mean, dataset_std = 0, 0
gen_channels = 3
disc_channels = 3
primary_loss_weight = 1
secondary_gen_loss = torch.nn.MSELoss()
secondary_gen_loss_weight = 0.8
std_loss = True
std_loss_weight = 0.5

#Hyper Parameters
epochs = 2
batch_size = 5
generator_lr = 0.00001
discriminator_lr = 0.00001
gen_betas = (0.5, 0.999)
disc_betas = (0.5, 0.999)
log_interval = 1

#The type of generator trained for (NormalGL, Displacement, Roughness, Metalness, or AO)
current_gen = "Roughness"

gen_path = f"Models/{current_gen}Generator.pt"
gen_state_dict_path = f"ModelsWStateDict/{current_gen}_SD.pt"
disc_path = f"Models/{current_gen}Discriminator.pt"

def save_state_dict(model_path, output_path):
    model = torch.load(model_path, map_location=device)
    torch.save(model.state_dict(), output_path)

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        device = "mps"
        torch.mps.empty_cache()
    else:
        device = "cpu"

    # get_dataset(training_data_info_url, training_data_info_output, training_raw_data_path, data_heading, data_filter, num_data_points, "Data/TrainingRawImages", "Data/TrainingImages", print_result=False)

    loader, dataset_mean, dataset_std = load_paired_data(num_data_points, os.getcwd()+"/"+training_data_path+"/Color", os.getcwd()+"/"+training_data_path+f"/{current_gen}", "Color_", f"{current_gen}_")

    generator, discriminator = train_gan(data_loader=loader, device=device, secondary_gen_loss=secondary_gen_loss, primary_loss_weight=primary_loss_weight, secondary_gen_loss_weight=secondary_gen_loss_weight, epochs=epochs, generator_lr=generator_lr, discriminator_lr=discriminator_lr, generator_betas=gen_betas, discriminator_betas=disc_betas, std_loss=std_loss, std_loss_weight=std_loss_weight, generator_channels=gen_channels, discriminator_channels=disc_channels, save_models=True, generator_path=gen_path, discriminator_path=disc_path, log_interval=log_interval)
    # generator, discriminator = load_gan(gen_path, disc_path, device)

    save_state_dict(gen_path, gen_state_dict_path)

    test_single_sample(sample_dir="Testing/TestTexture.png", generator=generator, gen_type=current_gen, device=device, display_plot=True, display_sample=True, save_plot=True, plot_dir=f"Testing/{current_gen}.png", print_tensor=True)