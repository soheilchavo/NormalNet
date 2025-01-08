from torchvision import transforms

#Takes a dataset as a folder of images and normalizes it

def normalize_sample(datapoint, mean, std, standalone=False):

    if standalone:
        datapoint = datapoint.float() / 255

    transform1 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    print(datapoint.shape)
    if datapoint.shape[0] == 4:
        datapoint = datapoint[:3, :, :]

    datapoint = transform1(datapoint)

    if len(datapoint.shape) == 3:
        datapoint = datapoint.unsqueeze(0)

    return datapoint

def normalize_data(dataset):

    mean = 0.0
    std = 0.0

    for datapoint in dataset:
        datapoint[0] = datapoint[0].float() / 255
        datapoint[1] = datapoint[1].float() / 255
        mean += datapoint[0].mean() + datapoint[1].mean()
        std += datapoint[0].std() + datapoint[1].std()

    mean /= len(dataset)*2
    std /= len(dataset)*2


    normalized_dataset = []

    for datapoint in dataset:
        normalized_dataset.append([normalize_sample(datapoint[0], mean, std), normalize_sample(datapoint[1], mean, std)])

    return normalized_dataset, mean, std

def unnormalize_tensor(sample, mean, std):
    return sample * std + mean