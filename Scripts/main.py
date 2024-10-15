#This Project was Created using Photogrammetry PBR's from ambientCG.com,
#licensed under the Creative Commons CC0 1.0 Universal License.

from data_collector import*

# data_info_request(url="https://ambientCG.com/api/v2/downloads_csv?method=PBRPhotogrammetry&type=Material&sort=Popular", output_directory="Data/training_data_info")

download_dataset(data_info_path="Data/training_data_info", data_file_path="Data/TrainingImages", data_filter = ["1K-JPG", "1K-PNG"])