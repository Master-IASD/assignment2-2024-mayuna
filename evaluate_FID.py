import os
import numpy as np
import torch
from torchvision import datasets, transforms
from scipy.linalg import sqrtm
from PIL import Image
import glob
from sklearn.neighbors import NearestNeighbors

# Helper function to calculate mean and covariance
def calculate_statistics(images):
    images = images / 255.0  # Normalize images to [0, 1]
    mean = np.mean(images, axis=0)
    cov = np.cov(images, rowvar=False)
    return mean, cov

# Calculate FID score
def calculate_fid(real_images, fake_images):
    mu_real, sigma_real = calculate_statistics(real_images)
    mu_fake, sigma_fake = calculate_statistics(fake_images)

    # FID formula
    ssdiff = np.sum((mu_real - mu_fake)**2.0)
    covmean = sqrtm(sigma_real.dot(sigma_fake))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
    return fid

# Load real MNIST dataset
transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
mnist_data = datasets.MNIST(root='data/MNIST', train=True, transform=transform, download=True)
real_images = np.array([np.array(mnist_data[i][0]).flatten() for i in range(10000)])  # Sample 10k for comparison

# Load generated samples
fake_images = []
for file in glob.glob("samples/*.png")[:10000]:  # Only use 10k samples
    img = Image.open(file).convert('L')
    img = np.array(img).flatten()
    fake_images.append(img)
fake_images = np.array(fake_images)

# Calculate FID
fid_score = calculate_fid(real_images, fake_images)
print("FID Score:", fid_score)








# Helper function for precision and recall
def precision_recall(real_images, fake_images, k=5):
    # Fit nearest neighbors on real images
    neigh_real = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(real_images)
    distances_real, _ = neigh_real.kneighbors(fake_images)
    
    # Precision: proportion of generated samples close to real samples
    precision = np.mean(distances_real.max(axis=1) <= np.percentile(distances_real, 5))

    # Fit nearest neighbors on generated images
    neigh_fake = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(fake_images)
    distances_fake, _ = neigh_fake.kneighbors(real_images)

    # Recall: proportion of real samples close to generated samples
    recall = np.mean(distances_fake.max(axis=1) <= np.percentile(distances_fake, 5))

    return precision, recall

# Calculate Precision and Recall
precision, recall = precision_recall(real_images, fake_images)
print("Precision:", precision)
print("Recall:", recall)