import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3
ADD_NOISE = True
NOISE_FACTOR = 0.2

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            
            nn.Tanh()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


model = ConvAutoencoder().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)


def add_gaussian_noise(images, noise_factor=0.2):
    noisy_images = images + noise_factor * torch.randn_like(images)
    return torch.clamp(noisy_images, -1.0, 1.0)

train_losses = []
test_losses = []

print("Starting training...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_train_loss = 0.0
    for images, _ in train_loader:
        images = images.to(DEVICE)
        inputs = add_gaussian_noise(images, NOISE_FACTOR) if ADD_NOISE else images        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * images.size(0)
        
    epoch_train_loss = running_train_loss / len(train_dataset)
    train_losses.append(epoch_train_loss)
    model.eval()
    running_test_loss = 0.0
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(DEVICE)
            inputs = add_gaussian_noise(images, NOISE_FACTOR) if ADD_NOISE else images        
            outputs = model(inputs)
            loss = criterion(outputs, images)
            running_test_loss += loss.item() * images.size(0)
            
    epoch_test_loss = running_test_loss / len(test_dataset)
    test_losses.append(epoch_test_loss)

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch [{epoch:02d}/{EPOCHS}] | Train Loss: {epoch_train_loss:.5f} | Test Loss: {epoch_test_loss:.5f}")

torch.save(model.state_dict(), "conv_autoencoder_cifar10.pth")
print("\nModel saved to 'conv_autoencoder_cifar10.pth'")

def denormalize(tensor):
    tensor = tensor.cpu().numpy().transpose((1, 2, 0))
    tensor = (tensor * 0.5) + 0.5
    return np.clip(tensor, 0, 1)

plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss", color="blue")
plt.plot(range(1, EPOCHS + 1), test_losses, label="Test Loss", color="red", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("MSE Reconstruction Loss")
plt.title("Convolutional Autoencoder Training & Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png")
plt.show()

model.eval()
data_iter = iter(test_loader)
images, _ = next(data_iter)
images = images[:8].to(DEVICE)
inputs = add_gaussian_noise(images, NOISE_FACTOR) if ADD_NOISE else images

with torch.no_grad():
    outputs = model(inputs)

num_images = 8
fig, axes = plt.subplots(3 if ADD_NOISE else 2, num_images, figsize=(15, 6 if ADD_NOISE else 4))

for i in range(num_images):
    if ADD_NOISE:
        axes[0, i].imshow(denormalize(inputs[i]))
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Noisy Input", loc="left", fontsize=10, fontweight="bold")
            
        axes[1, i].imshow(denormalize(outputs[i]))
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Reconstructed", loc="left", fontsize=10, fontweight="bold")
            
        axes[2, i].imshow(denormalize(images[i]))
        axes[2, i].axis("off")
        if i == 0:
            axes[2, i].set_title("Ground Truth", loc="left", fontsize=10, fontweight="bold")
    else:
        axes[0, i].imshow(denormalize(images[i]))
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original Input", loc="left", fontsize=10, fontweight="bold")
            
        axes[1, i].imshow(denormalize(outputs[i]))
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Reconstructed", loc="left", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("reconstructions.png")
plt.show()