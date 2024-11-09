import torch
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim

from model import Generator, Discriminator
from utils import D_train, G_train, save_models

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GAN on MNIST with Mode-Seeking Loss.')
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training.")
    parser.add_argument("--lr_generator", type=float, default=0.0002, help="Learning rate for the Generator.")
    parser.add_argument("--lr_discriminator", type=float, default=0.0002, help="Learning rate for the Discriminator.")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size for SGD")
    parser.add_argument("--mode_seeking_lambda", type=float, default=0.05, help="Lambda for mode-seeking loss")

    args = parser.parse_args()

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Dataset loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    # Model and optimizer setup
    mnist_dim = 784
    G = torch.nn.DataParallel(Generator(g_output_dim=mnist_dim)).cuda()
    D = torch.nn.DataParallel(Discriminator(d_input_dim=mnist_dim)).cuda()
    criterion = nn.BCELoss()
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr_generator)
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr_discriminator)

    print('Starting Training:')
    for epoch in trange(1, args.epochs + 1):
        G_loss_epoch = 0
        D_loss_epoch = 0
        for x, _ in train_loader:
            x = x.view(-1, mnist_dim)
            D_loss = D_train(x, G, D, D_optimizer, criterion)
            G_loss = G_train(x, G, D, G_optimizer, criterion, mode_seeking_lambda=args.mode_seeking_lambda)
            G_loss_epoch += G_loss
            D_loss_epoch += D_loss
            
        # Calculate and print average losses for the epoch
        avg_G_loss = G_loss_epoch / len(train_loader)
        avg_D_loss = D_loss_epoch / len(train_loader)
        print(f"Epoch [{epoch}/{args.epochs}] - Generator Loss: {avg_G_loss:.4f}, Discriminator Loss: {avg_D_loss:.4f}")

        # Save model checkpoints every 10 epochs
        if epoch % 10 == 0:
            save_models(G, D, 'checkpoints')

    print('Training complete.')
