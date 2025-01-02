import torch

def train(epochs, gen, disc, loader, loss_function, disc_optim, gen_optim, log_interval=0):
    for epoch in range(epochs):
        for batch_idx, (diffuse, real_normal) in enumerate(loader):

            #Generate the predicted normal
            fake_normal = gen(real_normal)

            #Determine the discriminator loss for the real normal map
            disc_real = disc(real_normal)
            loss_d_real = loss_function(disc_real, torch.ones_like(disc_real))

            #Determine the discriminator loss for the fake normal map
            disk_fake = disc(fake_normal)
            loss_d_fake = loss_function(disk_fake, torch.zeros_like(disk_fake))

            #Train discriminator based on the averaged fake and real loss
            loss_d = (loss_d_real + loss_d_fake)/2
            disc.zero_grad()
            loss_d.backward(retain_graph=True)
            disc_optim.step()

            #Train generator based on the discriminator feedback
            loss_g = loss_function(disk_fake, torch.ones_like(disk_fake))
            gen.zero_grad()
            loss_g.backward(retain_graph=True)
            gen_optim.step()

            #Log errors
            if log_interval > 0 and batch_idx % log_interval == 0:
                print(f"Epoch [{epoch + 1}/{epochs}] "
                      f"Batch [{batch_idx}/{len(loader)}] "
                      f"Loss_D: {loss_d.item():.4f} "
                      f"Loss_G: {loss_g.item():.4f}")