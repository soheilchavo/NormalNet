#This Project was Created using Photogrammetry PBR's from ambientCG.com,
#licensed under the Creative Commons CC0 1.0 Universal License.

#Imports
import os.path

import torch.optim
import pickle

import torchvision
from fontTools.merge.util import first
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from Data_Collection.data_collector import data_info_request, download_dataset
from Data_Collection.data_filtering import delete_duplicate_rows, filter_data, extract_dataset, pair_datapoints, transform_single_png
from Data_Collection.data_normalization import normalize_data, normalize_sample, scale_transform_sample

from Models.generator import UNet
from Models.discriminator import DiscriminatorCNN
from train import train_models
from test import single_pass, generate_pbr

#URL's and output directories for getting training and testing data (CSV of all materials and download links)
training_data_info_url = "https://ambientCG.com/api/v2/downloads_csv?method=PBRPhotogrammetry&type=Material&sort=Popular"
testing_data_info_url = ""
training_data_info_output = "Data/training_data_info"
testing_data_info_output = "Data/testing_data_info"
training_data_path = "Data/TrainingRawData"
testing_data_path = "Data/TestingRawData"

#Downloaded must have the following data types
data_filter = ["1K-PNG"]
data_heading = "downloadAttribute"
data_folders = ["AmbientOcclusion", "Color", "NormalDX", "NormalGL", "Roughness"]

#Parameters
num_data_points = 100
dataset_mean, dataset_std = 0, 0

#Hyper Parameters
epochs = 1
batch_size = 5
generator_lr = 0.0001
discriminator_lr = 0.003
beta1 = 0.6
beta2 = 0.999

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    # #Download, pair, and normalize dataset
    # data_info_request(url=training_data_info_url, output_directory=training_data_info_output)
    #
    # delete_duplicate_rows(csv_file_path=training_data_info_output)
    # filter_data(csv_file_path=training_data_info_output, data_heading=data_heading, data_filter=data_filter)
    #
    # download_dataset(data_info_path=training_data_info_output, data_file_path=training_data_path, data_filter = data_filter, num_data_points=num_data_points)
    # extract_dataset("Data/TrainingRawData", "Data/TrainingImages")

    # paired_dataset = pair_datapoints(num_data_points, os.getcwd()+"/Data/TrainingImages/Color", os.getcwd()+"/Data/TrainingImages/Roughness", "Color_", "Roughness_")
    #
    # normalized_data, dataset_mean, dataset_std = normalize_data(paired_dataset)

    # # #Save Training Data and Dataset Info
    # with open('Data/RoughnessTrainingData', 'wb') as f:
    #     pickle.dump(normalized_data, f)
    #
    # with open('Data/RoughnessTrainingDatasetInfo', 'wb') as f:
    #     pickle.dump([dataset_mean, dataset_std], f)
    #
    # #Loading code incase dataset is already saved
    # with open('Data/RoughnessTrainingData', 'rb') as f:
    #     dataset = pickle.load(f)
    #
    # with open('Data/RoughnessTrainingDatasetInfo', 'rb') as f:
    #     values = pickle.load(f)
    #
    # dataset_mean = values[0]
    # dataset_std = values[1]
    #
    # loader = DataLoader(dataset, shuffle=True)
    #
    # #Create models and optimizers
    # generator = UNet(3) #3 Channels for RGB
    # discriminator = DiscriminatorCNN(3) #3 Channels for RGB
    # generator_optim = torch.optim.Adam(generator.parameters(), lr=generator_lr, betas=(beta1, beta2))
    # discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr=discriminator_lr, betas=(beta1, beta2))
    #
    # train_models(epochs, generator, discriminator, loader, torch.nn.BCEWithLogitsLoss(), generator_optim, discriminator_optim, device, secondary_gen_loss= torch.nn.MSELoss(), secondary_loss_weight=0.3, log_interval=1)
    #
    # torch.save(generator, "Models/RoughnessGenerator.pt")
    # torch.save(discriminator, "Models/RoughnessDiscriminator.pt")

    # Test of a single sample

    #Transforms a png into a tensor for the model

    sample = transform_single_png("TestTexture.png")

    #Scales the tensor appropriately
    down_sample = scale_transform_sample(sample, standalone=True)

    strings = ["NormalGL", "Roughness", "Metalness", "Displacement", "AO"]

    generate_pbr(model_strings=strings, input_tensor=down_sample, guide_tensor=sample, device=device, save_plots=True, display_plots=True)

    #Load Generator
    # generator = torch.load("Models/RoughnessGenerator.pt")

    # single_pass(model=generator, input_tensor=down_sample, guide_tensor=sample, device=device, dataset_mean=dataset_mean, dataset_std=dataset_std, display_plot=True, display_sample=True, save_plot=True, plot_dir="Roughness.png", print_tensor=True)