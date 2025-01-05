from torchvision import transforms

#Takes a dataset as a folder of images and normalizes it
def normalize_data(dataset):

    mean = 0.0
    std = 0.0

    for datapoint in dataset:
        datapoint[0] = datapoint[0].float()
        datapoint[1] = datapoint[1].float()
        mean += datapoint[0].mean() + datapoint[1].mean()
        std += datapoint[0].std() + datapoint[1].std()

    mean /= len(dataset)*2
    std /= len(dataset)*2

    transform1 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((1024,1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    normalized_dataset = []

    for datapoint in dataset:
        normalized_dataset.append([transform1(datapoint[0]), transform1(datapoint[1])])

    return normalized_dataset
