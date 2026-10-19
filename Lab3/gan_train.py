import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

import os
import argparse

from gan import Generator, Discriminator

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 GAN Training')
parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
parser.add_argument('--ep', default=50, type=int, help='num of epochs')
args = parser.parse_args()

NOISE_DIM = 100

gen = Generator(NOISE_DIM)
dis = Discriminator()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gen.to(device)
dis.to(device)

criterion = nn.BCEWithLogitsLoss()
gen_optimiser = optim.Adam(gen.parameters(), lr = args.lr, betas=(0.5, 0.999))
dis_optimiser = optim.Adam(dis.parameters(), lr = args.lr, betas=(0.5, 0.999))

NUM_EPOCHS = args.ep
BATCH_SIZE = 256

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

def generate_and_save_images(model, epoch, noise):
    model.eval()
    with torch.no_grad():
        fake_images = model(noise).cpu()

    save_image(fake_images, f'test_images/image_at_epoch_{epoch+1:04d}.png',
               nrow=4, normalize=True, value_range=(-1, 1))

dis_losses, gen_losses = [], []
fixed_noise = torch.randn(16, NOISE_DIM, device=device)

for epoch in range(NUM_EPOCHS):
    dis_running, gen_running = 0.0, 0.0
    for i, data in enumerate(trainloader):
        real_images, _ = data
        real_images = real_images.to(device)

        dis_optimiser.zero_grad()
        real_labels = torch.ones(real_images.size(0), 1, 1, 1, device=device)
        real_outputs = dis(real_images)
        real_loss = criterion(real_outputs, real_labels)
        real_loss.backward()

        noise = torch.randn(real_images.size(0), NOISE_DIM, device=device)
        fake_images = gen(noise)
        fake_labels = torch.zeros(real_images.size(0), 1, 1, 1, device=device)
        fake_outputs = dis(fake_images.detach())
        fake_loss = criterion(fake_outputs, fake_labels)
        fake_loss.backward()
        dis_optimiser.step()

        gen_optimiser.zero_grad()
        fake_labels = torch.ones(real_images.size(0), 1, 1, 1, device=device)
        fake_outputs = dis(fake_images)
        gen_loss = criterion(fake_outputs, fake_labels)
        gen_loss.backward()
        gen_optimiser.step()

        dis_running += real_loss.item() + fake_loss.item()
        gen_running += gen_loss.item()

    dis_epoch = dis_running / len(trainloader)
    gen_epoch = gen_running / len(trainloader)
    dis_losses.append(dis_epoch)
    gen_losses.append(gen_epoch)
    print(f'Epoch {epoch+1}: D loss {dis_epoch:.4f}, G loss {gen_epoch:.4f}')

    if epoch + 1 in (1, 5) or (epoch + 1) % 10 == 0:
        generate_and_save_images(gen, epoch, fixed_noise)
        gen.train()

torch.save(dis.state_dict(), 'discriminator_final.pth')
torch.save(gen.state_dict(), 'generator_final.pth')

print(dis_losses)
print(gen_losses)