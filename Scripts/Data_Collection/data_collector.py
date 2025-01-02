import os
import requests

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
            print(f"{file_name}, Mirror link is corruped, datapoint not downloaded")
    if response != "":
        open(f"{data_directory}/{file_name}.zip", "wb").write(response.content)