import torch
from Models.generator import UNet
from Models.discriminator import DiscriminatorCNN
import matplotlib.pyplot as plt

generator_losses = []
discriminator_losses = []

def train_models(epochs, gen, disc, loader, loss_function, disc_optim, gen_optim, device, secondary_gen_loss = None, primary_loss_weight=1, secondary_loss_weight = 1, std_loss=False, std_loss_weight=0.5, log_interval=0, track_losses=True, print_losses=True, plot_losses=True):
    print("\nStarting Training...")

    #Load models to either CPU or GPU based on device
    gen = gen.to(device)
    disc = disc.to(device)

    for epoch in range(epochs):

        print(f"\nEpoch {epoch + 1}/{epochs}")
        for batch_idx, (diffuse, real_normal) in enumerate(loader):

            #Load images to CPU or GPU based on device
            diffuse = diffuse.to(device)
            real_normal = real_normal.to(device)

            #Tensors for training
            single_ones_tensor = torch.Tensor([1]).to(device)
            single_zeroes_tensor = torch.Tensor([0]).to(device)

            #Generate fake image
            fake_normal = gen(diffuse)

            #Calculate loss of generator based on discriminator feedback
            loss_g = primary_loss_weight*loss_function(disc(fake_normal), single_ones_tensor)

            #Add secondary loss function for generator if there is one
            if secondary_gen_loss:
                loss_g += secondary_loss_weight*secondary_gen_loss(fake_normal, real_normal)
            #Add a loss for std to maximize contrast
            if std_loss:
                std = torch.std(fake_normal.view(fake_normal.size(0), -1), dim=1)
                loss_g -= std_loss_weight * std.mean()

            #Update Generator gradient and take a step
            gen.zero_grad()
            loss_g.backward(retain_graph=True)
            gen_optim.step()

            #Guess the validity of real and fake images
            disc_real = disc(real_normal)
            disc_fake = disc(fake_normal)

            #Calculate discriminator loss
            loss_d_real = loss_function(disc_real, single_ones_tensor)
            loss_d_fake = loss_function(disc_fake, single_zeroes_tensor)
            loss_d = (loss_d_real + loss_d_fake)/2

            #Update Discriminator gradient and take a step
            disc.zero_grad()
            loss_d.backward(retain_graph=True)
            disc_optim.step()

            if device=="cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()

            if track_losses:
                generator_losses.append(loss_g.item())
                discriminator_losses.append(loss_d.item())

            #Log changes if on interval
            if log_interval > 0 and batch_idx % log_interval == 0:
                print(f"Batch [{batch_idx+1}/{len(loader)}] "
                      f"Loss_D: {loss_d:.4f} "
                      f"Loss_G: {loss_g:.4f}")

    print("\nTraining Finished.")

    if print_losses:
        print("Generator Losses:\n")
        for i in generator_losses:
            print(f"{i}")
        print("Discriminator Losses:\n")
        for i in discriminator_losses:
            print(f"{i}")
    if plot_losses:
        plt.title("Generator Loss [red] vs. Discriminator Loss [blue]")
        plt.xlabel("Bath #")
        plt.ylabel("Loss")
        plt.plot(generator_losses, color="red", label="Generator Loss")
        plt.plot(discriminator_losses, color="blue", label="Discriminator Loss")
        plt.show()


def train_gan(data_loader, device, secondary_gen_loss, primary_loss_weight, secondary_gen_loss_weight, epochs, generator_lr, discriminator_lr, generator_betas, discriminator_betas, std_loss=False, std_loss_weight=0, generator_channels=3, discriminator_channels=3, save_models=False, generator_path="", discriminator_path="", log_interval=1):

    generator = UNet(generator_channels)
    discriminator = DiscriminatorCNN(discriminator_channels)

    generator_optim = torch.optim.Adam(generator.parameters(), lr=generator_lr, betas=(generator_betas[0], generator_betas[1]))
    discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr=discriminator_lr, betas=(discriminator_betas[0], discriminator_betas[1]))

    train_models(epochs, generator, discriminator, data_loader, torch.nn.BCEWithLogitsLoss(), generator_optim,
                 discriminator_optim, device, primary_loss_weight=primary_loss_weight, secondary_gen_loss=secondary_gen_loss,
                 secondary_loss_weight=secondary_gen_loss_weight, std_loss=std_loss, std_loss_weight=std_loss_weight,
                 log_interval=log_interval)

    if save_models:
        torch.save(generator, generator_path)
        torch.save(discriminator, discriminator_path)

    return generator, discriminator