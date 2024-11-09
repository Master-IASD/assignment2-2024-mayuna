import torch
import torchvision
import os
import argparse

from model import Generator, Discriminator
from utils import load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate samples using trained GAN with Rejection Sampling and Latent Space Interpolation.')
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for generation.")
    parser.add_argument("--threshold", type=float, default=0.6, help="Rejection threshold for Discriminator confidence.")
    parser.add_argument("--interpolation", type=bool, default=True, help="Use latent space interpolation.")
    args = parser.parse_args()

    print('Loading model...')
    mnist_dim = 784
    model = Generator(g_output_dim=mnist_dim).cuda()
    model = load_model(model, 'checkpoints')
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    D = Discriminator(d_input_dim=mnist_dim).cuda()
    D = torch.nn.DataParallel(D).cuda()
    D.load_state_dict(torch.load('checkpoints/D.pth'))
    D.eval()

    print('Model loaded. Starting generation...')
    os.makedirs('samples', exist_ok=True)

    n_samples = 0
    batch_count = 0
    with torch.no_grad():
        while n_samples < 10000:
            batch_count += 1
            
            # Generate two sets of latent vectors
            z1 = torch.randn(args.batch_size, 100).cuda()
            z2 = torch.randn(args.batch_size, 100).cuda()
            
            if args.interpolation:
                # Interpolate between z1 and z2 for additional diversity
                alpha = torch.rand(args.batch_size, 1).cuda()  # Random interpolation factor
                z_interp = alpha * z1 + (1 - alpha) * z2  # Interpolated latent vector
                x = model(z_interp)  # Generate images from interpolated latent vectors
            else:
                x = model(z1)  # Generate images from the original latent vectors
            
            x = x.reshape(args.batch_size, 28, 28)
            accepted_in_batch = 0

            for k in range(x.shape[0]):
                if n_samples < 10000:
                    # Rejection Sampling: Keep samples with Discriminator confidence above threshold
                    score = D(x[k:k+1].view(-1, 784)).item()
                    if score > args.threshold:
                        torchvision.utils.save_image(x[k:k+1], os.path.join('samples', f'{n_samples}.png'))
                        n_samples += 1
                        accepted_in_batch += 1

            print(f"Batch {batch_count}: Generated {accepted_in_batch} samples (Total accepted: {n_samples}/10000)")

    print('Sample generation complete.')
