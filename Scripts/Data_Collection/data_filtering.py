import pandas
import zipfile
import os
from torchvision import transforms
from PIL import Image
import torch

img_transform = transforms.Compose([transforms.PILToTensor()])

def delete_duplicate_rows(csv_file_path : str, replace_file = True, output_path = None):
    df = pandas.read_csv(csv_file_path)
    df.drop_duplicates(subset=None, inplace=True)
    if replace_file:
        df.to_csv(csv_file_path, index=False)
    else:
        df.to_csv(output_path, index=False)

#Only keep datapoints with specific headings
def filter_data(csv_file_path : str, data_heading : str, data_filter : list[str], replace_file = True, output_path = None):
    df = pandas.read_csv(csv_file_path)
    df = df[df[data_heading].isin(data_filter)]
    if replace_file:
        df.to_csv(csv_file_path, index=False)
    else:
        df.to_csv(output_path, index=False)

#Extract zip files to different folders
def extract_maps(zip_path : str, folder_base_path : str, idx: int):
    with zipfile.ZipFile(zip_path, "r") as f:
        files = f.namelist()
        for name in f.namelist():
            try:
                file_type = name[name.index("PNG_")+4:name.index(".png")]
                new_filepath = f"{folder_base_path}/{file_type}"
                f.extract(name, path=new_filepath)
                os.rename(f"{new_filepath}/{name}", f"{new_filepath}/{file_type}_{idx}.png")
            except ValueError:
                pass
            except FileExistsError:
                pass

#Go through and extract each zip file
def extract_dataset(dataset_path: str, output_path : str):
    for idx, filename in enumerate(os.listdir(dataset_path)):
        f = os.path.join(dataset_path, filename)
        if f.endswith(".zip"):
            extract_maps(f, output_path, idx)

#Return a dictionary of corresponding datapoints
def pair_datapoints(n, folder1, folder2, prefix1, prefix2):
    out = []
    for i in range(n):
        datapoint1 = os.path.join(folder1, prefix1 + str(i) + ".png")
        datapoint2 = os.path.join(folder2, prefix2 + str(i) + ".png")

        if os.path.isfile(datapoint1) and os.path.isfile(datapoint2):
            data_tensor_1 = img_transform(Image.open(datapoint1))
            data_tensor_2 = img_transform(Image.open(datapoint2))

            if data_tensor_1.shape[0] == 3 and data_tensor_2.shape[0] == 3:
                if data_tensor_1.shape[1] == data_tensor_1.shape[2] and data_tensor_2.shape[1] == data_tensor_2.shape[2]:
                    out.append([data_tensor_1, data_tensor_2])
    return out

#Returns an image's corresponding tensor
def transform_single_png(sample):
    img = img_transform(Image.open(sample))
    dim = min(img.shape[1], img.shape[2])
    square_transform = transforms.Compose([transforms.Resize((dim, dim))])
    return square_transform(img)