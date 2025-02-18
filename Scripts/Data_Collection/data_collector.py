import os
import requests
import pickle
from torch.utils.data import DataLoader
from Scripts.Data_Collection.data_normalization import normalize_data
from Scripts.Data_Collection.data_filtering import delete_duplicate_rows, filter_data, extract_dataset, pair_datapoints

#Get CSV for list of datapoints in datasets
def data_info_request(url, output_directory, print_result : bool = False):
    print(f"Requesting data from {url}")
    response = requests.get(url)
    open(output_directory, "wb").write(response.content)
    print(f"Successful request, Data stored in {output_directory}")
    if print_result:
        print(response.text)

def download_dataset(data_info_path : str, data_file_path : str, data_filter : list[str], num_data_points=-1):
    if os.path.isfile(data_info_path):
        data_info = open(data_info_path, "rb")
        for i, line in enumerate(data_info):
            if i >= num_data_points: #If we've reached the cap on the number of datapoints we want to download
                return
            if i != 0: #Skip first line of CSV, the header
                line_string = str(line,'utf-8')
                line_array = line_string.split(",")
                if line_array[1] in data_filter:
                    link = line_array[4]
                    mirror_link = line_array[5]
                    download_datapoint(link, data_file_path, f"{line_array[0]}-{line_array[3]}", mirror_link)
    else:
        raise Exception("Data info file not included, run data_info_request() first.")

#Downloads a single zip file
def download_datapoint(link : str, data_directory : str, file_name : str, mirror_link : str = ""):
    print(f"Downloading {file_name} from {link}")
    response = ""
    try:
        response = requests.get(link)
    except:
        print(f"{file_name}, Primary link is corrupted, mirror link will be used.")
        try:
            response = requests.get(mirror_link)
        except:
            print(f"{file_name}, Mirror link is corrupted, datapoint not downloaded")
    if response != "":
        open(f"{data_directory}/{file_name}.zip", "wb").write(response.content)

def get_dataset(url, csv_file_path, data_file_path, data_heading, data_filter, num_data_points, raw_data_path, image_data_path, print_result = False):
    data_info_request(url=url, output_directory=csv_file_path, print_result=print_result)

    delete_duplicate_rows(csv_file_path=csv_file_path)
    filter_data(csv_file_path=csv_file_path, data_heading=data_heading, data_filter=data_filter)

    download_dataset(data_info_path=csv_file_path, data_file_path=data_file_path, data_filter = data_filter, num_data_points=num_data_points)
    extract_dataset(raw_data_path, image_data_path)

def save_dataset(data, mean, std, data_path, info_path):
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
    with open(info_path, 'wb') as f:
        pickle.dump([mean, std], f)

def load_paired_data(num_data_points, relative_data_path_1, relative_data_path_2, data_filter_1, data_filter_2, save_data=False, data_path="", info_path=""):

    paired_dataset = pair_datapoints(num_data_points, relative_data_path_1,
                                     relative_data_path_2, data_filter_1, data_filter_2)

    normalized_data, dataset_mean, dataset_std = normalize_data(paired_dataset)

    if save_data:
        save_dataset(normalized_data, dataset_mean, dataset_std, data_path, info_path)

    loader = DataLoader(normalized_data, shuffle=True)

    return loader, dataset_mean, dataset_std