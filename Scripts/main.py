#This Project was Created using Photogrammetry PBR's from ambientCG.com,
#licensed under the Creative Commons CC0 1.0 Universal License.
import os.path

import torch.optim
import pickle

from fontTools.merge.util import first
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from Data_Collection.data_collector import data_info_request, download_dataset
from Data_Collection.data_filtering import delete_duplicate_rows, filter_data, extract_dataset, pair_datapoints
from Data_Collection.data_normalization import normalize_data

from Models.generator import UNet
from Models.discriminator import DiscriminatorCNN
from train import train_models

#URL's and output directories for getting training and testing data (CSV of all materials and download links)
training_data_info_url = "https://ambientCG.com/api/v2/downloads_csv?method=PBRPhotogrammetry&type=Material&sort=Popular"
testing_data_info_url = ""
training_data_info_output = "Data/training_data_info"
testing_data_info_output = "Data/testing_data_info"

training_data_path = "Data/TrainingRawData"
testing_data_path = "Data/TestingRawData"

#Every file that is downloaded must have the following data types
data_filter = ["1K-PNG"]
data_heading = "downloadAttribute"

data_folders = ["AmbientOcclusion", "Color", "NormalDX", "NormalGL", "Roughness"]

num_data_points = 100

training_dataset_loader, testing_dataset_loader = None, None

epochs = 5
batch_size = 5

generator_lr = 0.0002
discriminator_lr = 0.0002
beta1 = 0.5
beta2 = 0.999

if __name__ == '__main__':

    #Download and filter dataset
    # data_info_request(url=training_data_info_url, output_directory=training_data_info_output)
    # delete_duplicate_rows(csv_file_path=training_data_info_output)
    # filter_data(csv_file_path=training_data_info_output, data_heading=data_heading, data_filter=data_filter)

    # download_dataset(data_info_path=training_data_info_output, data_file_path=training_data_path, data_filter = data_filter, num_data_points=num_data_points)

    #Extract dataset to different folders
    # extract_dataset("Data/TrainingRawData", "Data/TrainingImages")
    #
    # paired_dataset = pair_datapoints(num_data_points, os.getcwd()+"/Data/TrainingImages/Color", os.getcwd()+"/Data/TrainingImages/NormalDX", "Color_", "NormalDX_")
    #
    # normalized_data = normalize_data(paired_dataset)
    #
    # plt.imshow(normalized_data[0][0].permute(1,2,0).numpy())
    # plt.show()
    #
    #
    # with open('TrainingData', 'wb') as f:
    #     pickle.dump(normalized_data, f)

    with open('TrainingData', 'rb') as f:
        dataset = pickle.load(f)

    loader = DataLoader(dataset, shuffle=True)
    first_sample = next(iter(loader))

    generator = UNet(3) #3 Channels for RGB
    discriminator = DiscriminatorCNN(3) #3 Channels for RGB

    generator_optim = torch.optim.Adam(generator.parameters(), lr=generator_lr, betas=(beta1, beta2))
    discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr=discriminator_lr, betas=(beta1, beta2))

    train_models(epochs, generator, discriminator, loader, torch.nn.BCEWithLogitsLoss(), generator_optim, discriminator_optim, log_interval=1)