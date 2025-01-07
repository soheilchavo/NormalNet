import torch

def train_models(epochs, gen, disc, loader, loss_function, disc_optim, gen_optim, device, log_interval=0):
    print("\nTraining Started.")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        for batch_idx, (diffuse, real_normal) in enumerate(loader):

            diffuse = diffuse.to(device)
            real_normal = real_normal.to(device)

            gen = gen.to(device)
            disc = disc.to(device)

            single_ones_tensor = torch.Tensor([1]).to(device)
            single_zeroes_tensor = torch.Tensor([0]).to(device)

            fake_normal = gen(diffuse)

            loss_g = loss_function(disc(fake_normal), single_ones_tensor)
            gen.zero_grad()
            loss_g.backward(retain_graph=True)
            gen_optim.step()

            disc_real = disc(real_normal)

            disc_fake = disc(fake_normal)
            loss_d_real = loss_function(disc_real, single_ones_tensor)
            loss_d_fake = loss_function(disc_fake, single_zeroes_tensor)
            loss_d = (loss_d_real + loss_d_fake)/2

            disc.zero_grad()
            loss_d.backward(retain_graph=True)
            disc_optim.step()

            torch.cuda.empty_cache()

            if log_interval > 0 and batch_idx % log_interval == 0:
                print(f"Epoch [{epoch + 1}/{epochs}] "
                      f"Batch [{batch_idx}/{len(loader)}] "
                      f"Loss_D: {loss_d:.4f} "
                      f"Loss_G: {loss_g:.4f}")