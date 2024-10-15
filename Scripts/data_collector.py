import os
import requests

def data_info_request(url, output_directory, print_result : bool = False):
    response = requests.get(url)
    open(output_directory, "wb").write(response.content)
    if print_result:
        print(response.text)

def download_dataset(data_info_path : str, data_file_path : str, data_filter : list[str]):
    if os.path.isfile(data_info_path):
        try:
            data_info = open(data_info_path, "rb")
            for i, line in enumerate(data_info):
                if i == 3:
                    return
                if i != 0:
                    line_string = str(line,'utf-8')
                    line_array = line_string.split(",")
                    if line_array[1] in data_filter:
                        link = line_array[4]
                        mirror_link = line_array[5]
                        download_datapoint(link, data_file_path, f"{line_array[0]}-{line_array[3]}", mirror_link)
        except:
            raise Exception("Data info file does not exist or its formatting or is corrupted.")
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