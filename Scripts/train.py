import torch

def train_models(epochs, gen, disc, loader, loss_function, disc_optim, gen_optim, log_interval=0):
    print("\nTraining Started.")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        for batch_idx, (diffuse, real_normal) in enumerate(loader):
            print(f"Batch {batch_idx + 1}/{len(loader)}")

            fake_normal = gen(diffuse)

            loss_g = loss_function(disc(fake_normal), torch.Tensor([1]))
            gen.zero_grad()
            loss_g.backward(retain_graph=True)
            gen_optim.step()

            disc_real = disc(real_normal)

            disc_fake = disc(fake_normal)
            loss_d_real = loss_function(disc_real, torch.ones_like(disc_real))
            loss_d_fake = loss_function(disc_fake, torch.zeros_like(disc_fake))
            loss_d = (loss_d_real + loss_d_fake)/2

            disc.zero_grad()
            loss_d.backward(retain_graph=True)
            disc_optim.step()

            #Log errors
            # if log_interval > 0 and batch_idx % log_interval == 0:
            #     print(f"Epoch [{epoch + 1}/{epochs}] "
            #           f"Batch [{batch_idx}/{len(loader)}] "
            #           f"Loss_D: {loss_d.item():.4f} "
            #           f"Loss_G: {loss_g.item():.4f}")