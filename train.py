import torch
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim

from model import Generator, Discriminator
from utils import train_GAN, D_train, G_train, save_models

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CC-GAN with GMM and Feature Matching.')
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--feature_matching", action="store_true", help="Enable feature matching loss.")

    args = parser.parse_args()

    # Prepare folders for checkpoints and data
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Load MNIST dataset
    print('Loading MNIST dataset...')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print('Dataset loaded.')

    # Initialize models
    mnist_dim = 784
    G = torch.nn.DataParallel(Generator(g_output_dim=mnist_dim)).cuda()
    D = torch.nn.DataParallel(Discriminator(d_input_dim=mnist_dim, n_classes=10)).cuda()
    print('Model initialized.')

    # Set up loss functions and optimizers
    gan_criterion = nn.BCELoss()  # Binary cross-entropy for GAN real/fake discrimination
    class_criterion = nn.CrossEntropyLoss()  # Cross-entropy for class labels
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr)
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr)

    # Start training with epoch-wise loop and checkpoint saving
    print('Start Training :')
    n_epochs = args.epochs
    for epoch in trange(1, n_epochs + 1, leave=True):
        d_losses, g_losses = [], []
        
        for batch_idx, (x, y_class) in enumerate(train_loader):
            x = x.view(-1, mnist_dim).cuda()
            y_class = y_class.cuda()

            # Train Discriminator
            d_loss = D_train(x, y_class, G, D, D_optimizer, gan_criterion, class_criterion, feature_matching=args.feature_matching)
            d_losses.append(d_loss)

            # Train Generator
            g_loss = G_train(x, G, D, G_optimizer, gan_criterion, feature_matching=args.feature_matching)
            g_losses.append(g_loss)

        # Print average losses for each epoch
        avg_d_loss = sum(d_losses) / len(d_losses)
        avg_g_loss = sum(g_losses) / len(g_losses)
        print(f"Epoch [{epoch}/{n_epochs}], Discriminator Loss: {avg_d_loss:.4f}, Generator Loss: {avg_g_loss:.4f}")

        # Save model checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_models(G, D, 'checkpoints')
            print(f"Checkpoint saved at epoch {epoch}")

    print('Training done')
