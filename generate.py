import torch
import torchvision
import os
import argparse


from model import Generator
from model import Discriminator
from utils import load_model

def load_decoder(D, folder):
    ckpt = torch.load(os.path.join(folder,'D.pth'))
    D.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return D

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for training.")
    args = parser.parse_args()


    path = 'checkpoints'
    epsilon = 0.01


    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784

    model = Generator(g_output_dim = mnist_dim).cuda()
    model = load_model(model, 'checkpoints')
    model = torch.nn.DataParallel(model).cuda()
    model.eval()


    decoder = Discriminator(mnist_dim).cuda()
    decoder = load_decoder(decoder,path)
    decoder = torch.nn.DataParallel(decoder).cuda()
    decoder.eval()

    print('Model loaded.')

    #Calculating M
    z = torch.randn(10000, 100).cuda()
    x = model(z)
    output = decoder(x)
    D_bar = output/(1-output)
    M = torch.quantile(D_bar, 0.95)


    print('Start Generating')
    os.makedirs('samples', exist_ok=True)

    n_samples = 0
    with torch.no_grad():
        while n_samples<10000:
            z = torch.randn(args.batch_size, 100).cuda()
            x = model(z)
            output = decoder(x)
            D_bar = torch.log(output/(1-output))
            p = torch.sigmoid(D_bar - torch.log(M) - torch.log(1-torch.exp(D_bar-torch.log(M)-epsilon)))
            phi = torch.rand(args.batch_size)
            x = x.reshape(args.batch_size, 28, 28)
            for k in range(x.shape[0]):
                if n_samples<10000:
                    if phi[k] <= p[k]:
                        torchvision.utils.save_image(x[k:k+1], os.path.join('samples', f'{n_samples}.png'))
                        n_samples += 1




