import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 128

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, attr_names=None, max_variants=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.png')
        ]
        
        self.attr_names = attr_names
        self.max_variants = max_variants

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        csv_path = img_path.replace('.png', '.csv')
        attributes = self._load_csv_attributes(csv_path)

        attributes_tensor = torch.tensor(attributes, dtype=torch.long)

        return image, attributes_tensor

    def _load_csv_attributes(self, csv_path):
        attributes = []
        target_line = 12  # Numer linii do wczytania (numerujemy od 1)
        with open(csv_path, 'r') as f:
            for i, line in enumerate(f, start=1):
                if i == target_line or i == target_line + 1:
                    parts = line.strip().split(',')
                    attr_name = parts[0].strip('"')
                    variant, total_variants = map(int, parts[1:])
                    attributes.append(variant)
        return attributes


class Generator(nn.Module):
    def __init__(self, max_variants, embedding_dim):
        super(Generator, self).__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(num_embeddings, embedding_dim) for num_embeddings in max_variants])
        self.model = nn.Sequential(
            nn.Linear(latent_dim + len(max_variants) * embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh(),
        )

    def forward(self, z, attrs):
        attr_embeddings = [F.normalize(embedding(attrs[:, i]), p=2, dim=1) for i, embedding in enumerate(self.embeddings)]
        attrs_embed = torch.cat(attr_embeddings, dim=1)
        x = torch.cat((z, attrs_embed), dim=1)
        img = self.model(x)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, max_variants, embedding_dim):
        super(Discriminator, self).__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(num_embeddings, embedding_dim) for num_embeddings in max_variants])
        self.model = nn.Sequential(
            nn.Linear(int(torch.prod(torch.tensor(img_shape))) + len(max_variants) * embedding_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, attrs):
        attr_embeddings = [F.normalize(embedding(attrs[:, i]), p=2, dim=1) for i, embedding in enumerate(self.embeddings)]
        attrs_embed = torch.cat(attr_embeddings, dim=1)
        img_flat = img.view(img.size(0), -1)
        x = torch.cat((img_flat, attrs_embed), dim=1)
        validity = self.model(x)
        return validity
    
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def load_model(generator, discriminator, optimizer_G, optimizer_D, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Loaded model from epoch {epoch}")
        return epoch
    else:
        print("No checkpoint found, starting training from scratch.")
        return 0
    
attr_names = ["face_color", "hair_color"]
max_variants = [11, 10]

root_dir = "./cartoonset10k"
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])
dataset = CustomImageDataset(
    root_dir=root_dir,
    transform=transform,
    attr_names=attr_names,
    max_variants=max_variants
)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

latent_dim = 128
embedding_dim = 32
epochs = 200
attr_dim = len(attr_names)
img_shape = (3, image_size, image_size)
generator = Generator(max_variants, embedding_dim).to(device)
discriminator = Discriminator(max_variants, embedding_dim).to(device)
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
adversarial_loss = nn.BCELoss()

checkpoint_path = "models/gan_epoch_70.pth"  # Wskaż właściwą ścieżkę do zapisanego modelu
start_epoch = load_model(generator, discriminator, optimizer_G, optimizer_D, checkpoint_path)


def get_random_attributes_from_csv(dataset, num_samples=4):
    random_indices = random.sample(range(len(dataset)), num_samples)
    attributes = []
    filenames = []
    for idx in random_indices:
        img_path = dataset.image_paths[idx]
        filenames.append(os.path.basename(img_path))
        attributes.append(dataset[idx][1])
    return torch.stack(attributes), filenames

fixed_idxs = torch.randint(0, len(dataset), (4,))
fixed_samples = [dataset[i] for i in fixed_idxs]
fixed_attrs, fixed_filenames = get_random_attributes_from_csv(dataset, num_samples=4)
fixed_attrs = fixed_attrs.to(device)

def save_model(generator, discriminator, optimizer_G, optimizer_D, epoch, save_dir="models"):
    if epoch % 5 == 0:
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
        }, os.path.join(save_dir, f"gan_epoch_{epoch}.pth"))
        print(f"Model saved at epoch {epoch} to {save_dir}/gan_epoch_{epoch}.pth")

# Funkcja do wyświetlania oryginalnych i wygenerowanych obrazów
def show_generated_images(real_imgs, gen_imgs, attrs, filenames, epoch, device):
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    
    for i in range(4):
        # Oryginalne obrazy
        original_img_path = os.path.join(root_dir, filenames[i])
        original_img = Image.open(original_img_path).convert("RGB")
        original_img = transforms.ToTensor()(original_img).to(device)

        axes[0, i].imshow(original_img.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        axes[0, i].axis("off")
        axes[0, i].set_title(f"Oryginał: {filenames[i]}")

        # Wygenerowane obrazy
        axes[1, i].imshow(gen_imgs[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        axes[1, i].axis("off")
        axes[1, i].set_title(f"Wygenerowany: {filenames[i]}")

    plt.tight_layout()
    plt.show()

# Trening
for epoch in range(start_epoch,epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for i, (imgs, attrs) in enumerate(dataloader):
        real_imgs = imgs.to(device)
        attrs = attrs.to(device)
        valid = torch.ones((imgs.size(0), 1), device=device)
        fake = torch.zeros((imgs.size(0), 1), device=device)

        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs, attrs), valid)
        z = torch.randn((imgs.size(0), latent_dim), device=device)
        gen_imgs = generator(z, attrs)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), attrs), fake)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        g_loss = adversarial_loss(discriminator(gen_imgs, attrs), valid)
        g_loss.backward()
        optimizer_G.step()
        print(f"[Epoch {epoch+1}/{epochs}] [Batch {i+1}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

    print(f"[Epoch {epoch+1}/{epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

    if epoch % 5 == 0:
        with torch.no_grad():
            z = torch.randn((4, latent_dim), device=device)
            gen_imgs = generator(z, fixed_attrs).cpu()
            show_generated_images(real_imgs, gen_imgs, fixed_attrs, fixed_filenames, epoch, device)

    save_model(generator, discriminator, optimizer_G, optimizer_D, epoch)
