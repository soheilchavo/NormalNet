#This Project was Created using Photogrammetry PBR's from ambientCG.com,
#licensed under the Creative Commons CC0 1.0 Universal License.

from data_collector import*
from model import*

#URL's and output directories for getting training and testing data (CSV of all materials and download links)
training_data_info_url = "https://ambientCG.com/api/v2/downloads_csv?method=PBRPhotogrammetry&type=Material&sort=Popular"
testing_data_info_url = ""
training_data_info_output = "Data/training_data_info"
testing_data_info_output = "Data/testing_data_info"

training_data_path = "Data/TrainingImages"
testing_data_path = "Data/TestingImages"

#Every file that is downloaded must have the following data types
data_filter = ["1K-JPG", "1K-PNG"]

# data_info_request(url=training_data_info_url, output_directory=training_data_info_output)

# download_dataset(data_info_path=training_data_info_output, data_file_path=training_data_path, data_filter = data_filter)