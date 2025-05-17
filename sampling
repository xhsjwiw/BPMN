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



test_flag = True
img_size = 256
batch_size = 16
channel = 4
total = 1
save_model_dir = '/root/autodl-tmp/BPMN_MODEL/bpmn_Class.pth'
sr_model_dir = '/root/autodl-tmp/BPMN_MODEL/sr1.pth'
sample_number = 1
n_epochs = 150
total = 150
proportion = 0.5


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




def test_sample(Sample_number, Channel, Img_size,noise_schedulers,modle, proportion=0):
    # sample = torch.randn(Sample_number, Channel, Img_size, Img_size).to(device)
    # print(sample.shape)


    Mask_Img = np.zeros((1,4,img_size//8,img_size//8))
    if(proportion):
        width, height = img_size//8, img_size//8
        mask = Image.new('1', (width, height), 0)  # Initialize to all black

        for y in range(height):
            for x in range(width):
                if x < int(width * (1 - proportion)) :
                    mask.putpixel((x, y), 1)
        mask_np = np.array(mask)
        mask_img = mask_np.astype(np.uint8) * 255
        
    else:mask_img = Image.open('/root/autodl-tmp/B_277_333.jpg')
    
    mask_img = cv2.cvtColor(np.asarray(mask_img),cv2.COLOR_RGB2BGR)
    mask_img = cv2.resize(mask_img,(img_size//8, img_size//8))
    mask_img = Image.fromarray(cv2.cvtColor(mask_img,cv2.COLOR_BGR2RGB))
    mask_img = mask_img.convert('RGBA')
    r, g, b, a = mask_img.split()
    a = r.point(lambda x: x)
    mask_img.putalpha(a)
    print(mask_img)
    
    # mask_img = mask_img.convert('RGBA')
    # mask_img .save('/root/autodl-tmp/img_sample/B_118_111.png')
    # return
    
    # mask_img = mask_img.resize((img_size,img_size),Image.LANCZOS)
    # plt.imshow(mask_img)
    # plt.show()
    
    
    # mask_img = cv2.cvtColor(np.asarray(mask_img),cv2.COLOR_RGB2BGR)
    # mask_img = cv2.resize(mask_img,(img_size//8, img_size//8))
    # mask_img = Image.fromarray(cv2.cvtColor(mask_img,cv2.COLOR_BGR2RGB))

    # mask_img = mask_img.resize((img_size//8,img_size//8),Image.LANCZOS)
    
    mask_img = np.array(mask_img)
    mask_img = torch.tensor(mask_img)
    mask_img = mask_img.permute(2,0,1)
    Mask_Img[0,:,:] = mask_img
    Mask_Img = np.tile(Mask_Img,(1,1,1,1))
    Mask_Img = torch.tensor(Mask_Img)
    Mask_Img = torch.clamp(Mask_Img,0,1)
    DMask_Img = 1-Mask_Img
    Mask_Img = Mask_Img.to(torch.float32)
    DMask_Img = DMask_Img.to(torch.float32)
    Mask_Img = Mask_Img.to(device)
    DMask_Img = DMask_Img.to(device)
    
    # print(Mask_Img)
    # print(DMask_Img)
    
    X = show_images(Mask_Img)
    plt.imshow(X)
    plt.show()
    
    X = show_images(DMask_Img)
    plt.imshow(X)
    plt.show()
    # return
    
    

    img = Image.open('/root/autodl-tmp/image/对比/原图/267.jpg')
    x = np.zeros((1,img_size,img_size,3),dtype=np.uint8)
    y = np.zeros((1,3,img_size,img_size))
    x[0,:,:] = img.resize((img_size,img_size),Image.LANCZOS)
    X = train_transform(x[0])
    y[0,:,:] = X
    y = np.tile(y,(1,1,1,1))
    sample = torch.tensor(y)
    sample = sample.to(torch.float32)
    sample = sample.to(device)

    text = ""
    # text = "Structured BPMN, complete structure, have start and end flags, four gateway, thirteen activities"
    # text = "Structured BPMN, complete structure, have start and end flags, four gateway, fifteen activities"
    # text = "Structured BPMN, incomplete structure, no start and end flags, four gateway, thirteen activities"
    text = "Structured BPMN, complete structure, have start and end flags, four gateway, fifteen activities"
    X = show_images(sample)
    plt.imshow(X)
    plt.show()
    
    
    with torch.no_grad():
        sample = vae.encode(sample).latent_dist.sample()
        sample = sample * vae.config.scaling_factor # rescaling latents
        
        X = show_images(sample)
        plt.imshow(X)
        plt.show()
        
        
        text_input = tokenizer(text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

    clean_images = sample.to(device)
    noise = torch.randn(clean_images.shape).to(clean_images.device)
    bs = clean_images.shape[0]

    timesteps = torch.randint(4999, 5000, (bs,), device=clean_images.device).long()
    m = noise_schedulers.add_noise(clean_images, noise, timesteps)
    m = m.to(torch.float32)

    a_1 = noise_schedulers.alphas_cumprod ** 0.5
    a_2 = (1 - noise_schedulers.alphas_cumprod) ** 0.5

    for t in range(4999,0,-1):   
      if(t>1500):   
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        timesteps = torch.randint(t, t+1, (bs,), device=clean_images.device).long()
        noisy_images = noise_schedulers.add_noise(clean_images, noise, timesteps)

        # with torch.no_grad():
        #     residual_M = modle(noisy_images, t, encoder_hidden_states = T).sample
        #     residual = modle(m, t, encoder_hidden_states = text_embeddings).sample
        # noisy_images = noise_schedulers.step(residual_M, t, noisy_images).prev_sample
        # m = noise_schedulers.step(residual, t, m).prev_sample
                
        m_1 = (Mask_Img*noisy_images).to(torch.float32)
        m_2 = (DMask_Img*m).to(torch.float32)
        m = m_1 + m_2
        # m = a_1[t]/a_1[t-1]*m + (a_2[t]-a_1[t]*a_2[t-1]/a_1[t-1])*noise
        for u in range(10):
            if(u<9):
               with torch.no_grad():
                   residual = modle(m, t, encoder_hidden_states = text_embeddings).sample
               m = noise_schedulers.step(residual, t, m).prev_sample
               m = a_1[t]/a_1[t-1]*m + (a_2[t]-a_1[t]*a_2[t-1]/a_1[t-1])*noise
               m = m.to(torch.float32)
            else:
               with torch.no_grad():
                   residual = modle(m, t, encoder_hidden_states = text_embeddings).sample
               m = noise_schedulers.step(residual, t, m).prev_sample
      else:
         with torch.no_grad():
             residual = modle(m, t, encoder_hidden_states = text_embeddings).sample
         m = noise_schedulers.step(residual, t, m).prev_sample 
        
      
      if(t%1000 == 0 or t == 1 or t==4999):
            print(t)
            with torch.no_grad():
                M = (1 / vae.config.scaling_factor) * m
                M = vae.decode(M).sample
            
            z = show_images(noisy_images)
            plt.imshow(z)
            plt.axis('off')
            plt.show()
            
            
#             z = show_images(m_1)
#             plt.imshow(z)
#             plt.show()
            
#             z = show_images(m_2)
#             plt.imshow(z)
#             plt.show()
            
#             z = show_images(m)
#             plt.imshow(z)
#             plt.show()
            
            z = show_images(M)
            plt.imshow(z)
            plt.axis('off')
            plt.show()
            
    with torch.no_grad():
        sample = sr(M, 0).sample 
        sample = sr(sample, 0).sample
    X = show_images(sample)
    plt.imshow(X)
    plt.axis('off')
    # plt.savefig('/root/autodl-tmp/image208/pre_image/pre_208_0.3.jpg')
    plt.savefig('/root/autodl-tmp/image/对比/diffusion/267.jpg', bbox_inches='tight', pad_inches=0)
    # plt.show()


def main():
    # train(net, train_dataloader, start_epoch, n_epochs, noise_scheduler)
    # test_sample(sample_number, channel, img_size, noise_scheduler, net)
    if test_flag:
        print("Load the trained model")
        # Load the saved model directly for testing and validation, skipping the steps that follow this module.
        checkpoint = torch.load(save_model_dir)
        unet.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        
        sr_checkpoint = torch.load(sr_model_dir)
        sr.load_state_dict(sr_checkpoint['model'])
        sr_optimizer.load_state_dict(sr_checkpoint['optimizer']) 
        
        test_sample(sample_number, channel, img_size, noise_scheduler, unet, 0.5)
        return



if __name__ == '__main__':
    main()
