import torch
import os

def D_train(x, G, D, D_optimizer, criterion):
    D.zero_grad()

    # Train on real images
    x_real, y_real = x, torch.ones(x.size(0), 1)
    x_real, y_real = x_real.cuda(), y_real.cuda()
    D_output_real = D(x_real)
    D_real_loss = criterion(D_output_real, y_real)

    # Train on fake images
    z = torch.randn(x.size(0), 100).cuda()
    x_fake, y_fake = G(z), torch.zeros(x.size(0), 1).cuda()
    D_output_fake = D(x_fake)
    D_fake_loss = criterion(D_output_fake, y_fake)

    # Combine losses
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
    
    return D_loss.item()

def G_train(x, G, D, G_optimizer, criterion, mode_seeking_lambda=0.05):
    G.zero_grad()

    # Standard Generator loss
    z = torch.randn(x.size(0), 100).cuda()
    y = torch.ones(x.size(0), 1).cuda()
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # Mode-Seeking Loss
    z1 = torch.randn(x.size(0), 100).cuda()
    z2 = z1 + torch.normal(0, 0.1, size=z1.size()).cuda()  # Slightly perturbed
    G_output1 = G(z1)
    G_output2 = G(z2)
    mode_seeking_loss = torch.mean((G_output1 - G_output2).pow(2).sum(dim=1))

    # Combine both losses
    total_G_loss = G_loss + mode_seeking_lambda * mode_seeking_loss
    total_G_loss.backward()
    G_optimizer.step()
    
    return total_G_loss.item()

def save_models(G, D, folder):
    os.makedirs(folder, exist_ok=True)
    torch.save(G.state_dict(), os.path.join(folder, 'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder, 'D.pth'))

def load_model(G, folder):
    checkpoint = torch.load(os.path.join(folder, 'G.pth'))
    G.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})
    return G
