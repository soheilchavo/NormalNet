import pandas
import zipfile
import os

def delete_duplicate_rows(csv_file_path : str, replace_file = True, output_path = None):
    df = pandas.read_csv(csv_file_path)
    df.drop_duplicates(subset=None, inplace=True)
    if replace_file:
        df.to_csv(csv_file_path, index=False)
    else:
        df.to_csv(output_path, index=False)

def filter_data(csv_file_path : str, data_heading : str, data_filter : list[str], replace_file = True, output_path = None):
    df = pandas.read_csv(csv_file_path)
    df = df[df[data_heading].isin(data_filter)]
    if replace_file:
        df.to_csv(csv_file_path, index=False)
    else:
        df.to_csv(output_path, index=False)

def extract_maps(zip_path : str, folder_base_path : str, folders : list[str]):
    with zipfile.ZipFile(zip_path, "r") as f:
        for name in f.namelist():
            for file_type in folders:
                if name.__contains__(file_type):
                    f.extract(name, path=f"{folder_base_path}/{file_type}")
                    break

def extract_dataset(dataset_path: str, output_path : str, folders : list[str]):
    for filename in os.listdir(dataset_path):
        f = os.path.join(dataset_path, filename)
        extract_maps(f, output_path, folders)