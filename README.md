![Normal Net](https://github.com/user-attachments/assets/8bd8f983-1eab-4409-948d-fece8f4ea555)

## 🔍 Description
Normal Net is a project that is designed to transform diffuse images into PBR Maps, currently focused on Normal Maps. Models are trained using PyTorch and a GAN archetecture. The project is being created in order to allow for 3D artists to quickly create photorealistic renders without having to download and store many PBR Materials locally. Currently a Blender Plugin is being created to allow users to seamlessly incorprate the technology into the work.

## 🛠️ Usage

If you want to pass a single sample into a pre-existing model, run the following code (Note that the dataset mean and standard deviation need to be passed in order to un-normalize the result from the generator):
```py
# Test of a single sample

#Transforms a png into a tensor for the model
sample = transform_single_png("TestTexture.png")
#Scales the tensor appropriately 
sample = scale_transform_sample(sample, standalone=True)

#Load Generator
generator = torch.load("Generator.pt")

single_pass(model=generator, input_tensor=sample, device=device, dataset_mean=dataset_mean, dataset_std=dataset_std, display_plot=True)
```

If you want to download the recommended dataset and save it locally, run the following code and change the parameters to match your local directories:

```py
#Download, pair, and normalize dataset
data_info_request(url=training_data_info_url, output_directory=training_data_info_output)

delete_duplicate_rows(csv_file_path=training_data_info_output)
filter_data(csv_file_path=training_data_info_output, data_heading=data_heading, data_filter=data_filter)

download_dataset(data_info_path=training_data_info_output, data_file_path=training_data_path, data_filter = data_filter, num_data_points=num_data_points)
extract_dataset("Data/TrainingRawData", "Data/TrainingImages")

paired_dataset = pair_datapoints(num_data_points, os.getcwd()+"/Data/TrainingImages/Color", os.getcwd()+"/Data/TrainingImages/NormalDX", "Color_", "NormalDX_")
normalized_data, dataset_mean, dataset_std = normalize_data(paired_dataset)

#Save Training Data and Dataset Info
with open('Data/TrainingData', 'wb') as f:
    pickle.dump(normalized_data, f)

with open('Data/TrainingDatsetInfo', 'wb') as f:
    pickle.dump([dataset_mean, dataset_std], f)
```

If you want to train new models, run the following code:

```py
loader = DataLoader(normalized_data, shuffle=True)

#Create models and optimizers
generator = UNet(3) #3 Channels for RGB
discriminator = DiscriminatorCNN(3) #3 Channels for RGB
generator_optim = torch.optim.Adam(generator.parameters(), lr=generator_lr, betas=(beta1, beta2))
discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr=discriminator_lr, betas=(beta1, beta2))

train_models(epochs, generator, discriminator, loader, torch.nn.BCEWithLogitsLoss(), generator_optim, discriminator_optim, device, secondary_gen_loss= torch.nn.MSELoss(), secondary_loss_weight=0.3, log_interval=1)

torch.save(generator, "Models/Generator.pt")
torch.save(discriminator, "Models/Discriminator.pt")
```

## 🚧 Feature Roadmap

- ✨ Size Generalization (Handling any size image)
- ✨ Bump, Displacement, Roughness, and Metallic Generators
- ✨ Blender Plugin
