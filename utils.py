import torch
import os
import numpy as np
from sklearn.mixture import GaussianMixture

latent_dim = 100
n_components = 10
gmm = GaussianMixture(n_components=n_components)
gmm.fit(np.random.randn(10000, latent_dim))  # Fit once on a large random sample

def sample_gmm_latent(batch_size):
    z = gmm.sample(batch_size)[0]
    return torch.tensor(z, dtype=torch.float32).cuda()

def D_train(x, y_class, G, D, D_optimizer, gan_criterion, class_criterion, feature_matching=False):
    D.zero_grad()

    # Train on real samples
    x_real, y_real = x, torch.ones(x.shape[0], 1).cuda()
    real_fake_output, class_output = D(x_real)
    D_real_loss = gan_criterion(real_fake_output, y_real)
    D_class_loss = class_criterion(class_output, y_class)

    # Train on fake samples
    z = sample_gmm_latent(x.shape[0])
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).cuda()
    fake_output, _ = D(x_fake)
    D_fake_loss = gan_criterion(fake_output, y_fake)
    
    # Combine losses
    D_loss = D_real_loss + D_fake_loss + D_class_loss

    if feature_matching:
        real_features = real_fake_output.mean(0)
        fake_features = fake_output.mean(0)
        feature_loss = torch.mean((real_features - fake_features) ** 2)
        D_loss += feature_loss

    # Update Discriminator
    D_loss.backward()
    D_optimizer.step()

    return D_loss.item()

def G_train(x, G, D, G_optimizer, gan_criterion, feature_matching=False):
    G.zero_grad()

    z = sample_gmm_latent(x.shape[0])
    y_real = torch.ones(x.shape[0], 1).cuda()
    G_output = G(z)
    fake_output, _ = D(G_output)

    G_loss = gan_criterion(fake_output, y_real)

    if feature_matching:
        real_features = fake_output.mean(0)
        fake_features = D(G_output)[0].mean(0)
        feature_loss = torch.mean((real_features - fake_features) ** 2)
        G_loss += feature_loss

    # Update Generator
    G_loss.backward()
    G_optimizer.step()

    return G_loss.item()

def train_GAN(epochs, G, D, G_optimizer, D_optimizer, train_loader, gan_criterion, class_criterion, feature_matching=False):
    for epoch in range(1, epochs + 1):
        d_losses, g_losses = [], []
        for batch_idx, (x, y_class) in enumerate(train_loader):
            x = x.view(-1, 784).cuda()
            y_class = y_class.cuda()

            # Train Discriminator
            d_loss = D_train(x, y_class, G, D, D_optimizer, gan_criterion, class_criterion, feature_matching)
            d_losses.append(d_loss)

            # Train Generator
            g_loss = G_train(x, G, D, G_optimizer, gan_criterion, feature_matching)
            g_losses.append(g_loss)

        # Print results for each epoch
        avg_d_loss = np.mean(d_losses)
        avg_g_loss = np.mean(g_losses)
        print(f"Epoch [{epoch}/{epochs}], Discriminator Loss: {avg_d_loss:.4f}, Generator Loss: {avg_g_loss:.4f}")


def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))


def load_model(G, folder):
    ckpt = torch.load(os.path.join(folder,'G.pth'))
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G