from datasets import load_from_disk
from matplotlib import pyplot as plt
import os
import pandas as pd
from torch.utils.data import DataLoader,Dataset
import time
import torchvision.transforms as transforms
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from diffusers import DDPMScheduler, UNet2DModel
from tqdm import tqdm,trange
from PIL import Image
from diffusers import AutoencoderKL, UNet2DModel, UNet2DConditionModel, DDPMScheduler,AsymmetricAutoencoderKL,ConsistencyDecoderVAE
from transformers import CLIPTextModel, CLIPTokenizer
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')



test_flag = False
img_size = 256
batch_size = 16
channel = 4
total = 1
save_model_dir = '/root/autodl-tmp/BPMN_MODEL/bpmn_Class.pth'
sr_model_dir = '/root/autodl-tmp/BPMN_MODEL/sr1.pth'
sample_number = 1
n_epochs = 1


def show_images(x):
    """Given a batch of images, create a grid and convert it to PIL"""
    x = x * 0.5 + 0.5  # Map the interval (-1, 1) back to the interval (0, 1)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im


def make_grid(images, size=img_size):
    output_im = Image.new("RGB", (size * len(images), size))
    for i, im in enumerate(images):
        output_im.paste(im.resize((size, size)), (i * size, 0))
    return output_im


train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),#Convert the image to a Tensor and normalize it
    transforms.Normalize([0.5],0.5),
    # transforms.Grayscale(num_output_channels=1),
])


#Dataset class handling
class ImgDataset(Dataset):
    def __init__(self,x,y = None,transform = None):
        self.x = x
        self.y = y
        # if y is not None:
        #     self.y = torch.tensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self,index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None: 
            Y = self.y[index]
            return X,Y 
        else:
            return X 
        
        
        
class ImgDataset1(Dataset):
    def __init__(self,x,y = None,transform = None):
        self.x = x
        self.y = y
        # if y is not None:
        #     self.y = torch.tensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self,index):
        X = self.x[index]
        Y = self.y[index]
        if self.transform is not None:
            X = self.transform(X)
            Y = self.transform(Y)
        if self.y is not None: 
            return X,Y 
        else:
            return X 


train_x = np.load('/root/autodl-tmp/bpmndataset/train_x_BPMN_256.npy')
train_y = np.load('/root/autodl-tmp/bpmndataset/train_y_BPMN_256.npy')
if(not test_flag):
    train_x = np.tile(train_x,(20,1,1,1))
    train_y = np.tile(train_y,(20,))



Dataset = ImgDataset(train_x, train_y, train_transform)
train_dataloader = DataLoader(Dataset, batch_size = batch_size, shuffle = True, drop_last = True)

it = next(iter(train_dataloader))
x, y = it
print(y[0])
X = show_images(x[0])
plt.imshow(X)
plt.show()



# Load autoencoder
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
# vae = AsymmetricAutoencoderKL.from_pretrained("cross-attention/asymmetric-autoencoder-kl-x-1-5")#-------------------------------
# vae = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder")
# Load text encoder
text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")
tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
# Initialize UNet
unet = UNet2DConditionModel(
    sample_size=img_size//8,  # the target image resolution
    in_channels=channel,  # the number of input channels, 3 for RGB images
    out_channels=channel,  # the number of output channels
    down_block_types=(
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    ),
    mid_block_type="UNetMidBlock2DCrossAttn",
    up_block_types=(
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D"),
    only_cross_attention=False,
    block_out_channels=(128, 256, 512, 512),
    layers_per_block=2,
    attention_head_dim=8,
    cross_attention_dim=768,
)  # <<<


sr = UNet2DModel(
    sample_size=256,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(64, 128, 128, 256),  # Roughly matching our basic unet example
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",  # a regular ResNet downsampling block
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",   # a regular ResNet upsampling block
        "UpBlock2D",
      ),
) #<<<
sr.to(device)
sr_optimizer = torch.optim.AdamW(sr.parameters(), lr=0.0001)


vae.requires_grad_(False)
# vae2.requires_grad_(False)
text_encoder.requires_grad_(False)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.AdamW(unet.parameters(), lr=0.0001)
losses = []
Loss = []
# Define scheduler
# noise_scheduler = DDPMScheduler(
#     beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
# )
noise_scheduler = DDPMScheduler(
    num_train_timesteps=5000, beta_schedule="squaredcos_cap_v2"
)

vae.to(device, dtype=torch.float32)
# vae2.to(device, dtype=torch.float32)
text_encoder.to(device, dtype=torch.float32)
unet = unet.to(device, dtype=torch.float32)


n_epochs = 150
total = 50
test_flag = True


def train(model, train_dataset, s_epoch, n_epoch, noise_schedulers):
    pbar = tqdm(total=total, desc="train")
    for epoch in range(s_epoch,n_epoch):
        n = 0
        # for batch in tqdm(train_dataset):
        for batch in train_dataset:

          n = n + 1

          x,y = batch
          x = x.to(device, dtype=torch.float32)
        
        
          with torch.no_grad():
            latents = vae.encode(x).latent_dist.sample()
            latents = latents * vae.config.scaling_factor # rescaling latents
            text_input = tokenizer(y, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

          noise = torch.randn_like(latents)
          bsz = latents.shape[0]
          timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
          timesteps = timesteps.long()
          noisy_latents = noise_schedulers.add_noise(latents, noise, timesteps)
          model_pred = model(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample


#           with torch.no_grad():
#             # latents = vae.encode(x).latent_dist.sample()
#             # latents = latents * vae.config.scaling_factor # rescaling latents
#             text_input = tokenizer(y, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
#             text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

#           noise = torch.randn_like(x)
#           bsz = x.shape[0]
#           timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=x.device)
#           timesteps = timesteps.long()
#           noisy_x = noise_schedulers.add_noise(x, noise, timesteps)
#           model_pred = model(noisy_x, timesteps, encoder_hidden_states=text_embeddings).sample
            
          loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")/2 
          loss.backward(loss)
          if n%2 == 0:
            n = 0
            optimizer.step()
            optimizer.zero_grad()
          losses.append(loss.item())


        pbar.update(1)
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, save_model_dir)


def main():
    # train(net, train_dataloader, start_epoch, n_epochs, noise_scheduler)
    # test_sample(sample_number, channel, img_size, noise_scheduler, net)
    if test_flag:
        print("Load the trained model")
        # Load the saved model directly for testing and validation, skipping the steps that follow this module
        checkpoint = torch.load(save_model_dir)
        unet.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        
        sr_checkpoint = torch.load(sr_model_dir)
        sr.load_state_dict(sr_checkpoint['model'])
        sr_optimizer.load_state_dict(sr_checkpoint['optimizer'])     
        
        
        test_sample(sample_number, channel, img_size, noise_scheduler, unet)
        return

    if os.path.exists(save_model_dir):
        checkpoint = torch.load(save_model_dir)
        unet.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        
        sr_checkpoint = torch.load(sr_model_dir)
        sr.load_state_dict(sr_checkpoint['model'])
        sr_optimizer.load_state_dict(sr_checkpoint['optimizer']) 
        
        print('Successfully loaded epoch {}!'.format(start_epoch))
    else:
        start_epoch = 0
        print('No saved model, training will start from scratch!')

    train(unet, train_dataloader, start_epoch, n_epochs, noise_scheduler)
    test_sample(sample_number, channel, img_size, noise_scheduler, unet)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss_ge')
    plt.show()
    plt.savefig('/root/BPMN_MODEL/loss')



if __name__ == '__main__':
    main()
