# -*- coding = utf-8 -*-
# @Time : 1/20/25 10:17
# @Author : Tracy
# @File : wtf.py
# @Software : PyCharm

import torch
import torch.optim as optim
from tqdm import tqdm

class TCNGANTrainer:
    def __init__(self, generator, discriminator, dataloader, nz, lr, clip, generator_path, file_name, device):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.dataloader = dataloader
        self.nz = nz  # Latent vector size
        self.clip = clip
        self.generator_path = generator_path
        self.file_name = file_name
        self.device = device
        self.gen_optimizer = optim.RMSprop(self.generator.parameters(), lr=lr)
        self.disc_optimizer = optim.RMSprop(self.discriminator.parameters(), lr=lr)

    def train_one_epoch(self, epoch):
        self.generator.train()
        self.discriminator.train()

        t = tqdm(self.dataloader, desc=f"Epoch {epoch}")
        for idx, data in enumerate(t):
            real = data.to(self.device)
            batch_size, seq_len = real.size(0), real.size(1)

            ### Train Discriminator ###
            self.discriminator.zero_grad()
            noise = torch.randn(batch_size, self.nz, seq_len, device=self.device)
            fake = self.generator(noise).detach()
            disc_loss = -torch.mean(self.discriminator(real)) + torch.mean(self.discriminator(fake))
            disc_loss.backward()
            self.disc_optimizer.step()

            # Apply weight clipping
            for dp in self.discriminator.parameters():
                dp.data.clamp_(-self.clip, self.clip)

            ### Train Generator (every 5 steps) ###
            if idx % 5 == 0:
                self.generator.zero_grad()
                noise = torch.randn(batch_size, self.nz, seq_len, device=self.device)
                gen_loss = -torch.mean(self.discriminator(self.generator(noise)))
                gen_loss.backward()
                self.gen_optimizer.step()

            # Update progress bar
            t.set_postfix(disc_loss=disc_loss.item(), gen_loss=gen_loss.item())

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.train_one_epoch(epoch)

            torch.save(
                self.generator,
                f"{self.generator_path}trained_generator_{self.file_name}_epoch_{epoch}.pth"
            )







