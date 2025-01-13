from torchvision import transforms

def scale_transform_sample(datapoint, standalone=False):

    #Standalone is true if the function is called on a single image rather than as a part of a dataset
    if standalone:
        datapoint = datapoint.float() / 255

    #Intial Size transform (256, 256) in order to save on processing, images get upscaled later
    transform1 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])

    #Removes alpha channel if image has one
    if datapoint.shape[0] == 4:
        datapoint = datapoint[:3, :, :]

    #Size transform
    datapoint = transform1(datapoint)

    #Add batch dimension if single sample doesen't have it
    if len(datapoint.shape) == 3:
        datapoint = datapoint.unsqueeze(0)

    return datapoint

def normalize_sample(datapoint, mean, std):

    normal_transform = transforms.Compose([
        transforms.Normalize(mean=mean, std=std)
    ])

    return normal_transform(datapoint)

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
        scaled_datapoints = [scale_transform_sample(datapoint[0]), scale_transform_sample(datapoint[1])]
        normalized_dataset.append([normalize_sample(scaled_datapoints[0], mean, std), normalize_sample(scaled_datapoints[1], mean, std)])

    return normalized_dataset, mean, std

def unnormalize_tensor(sample, mean, std):
    return sample * std + mean