
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, latent_dim, im_size, im_channels, conv_channels, kernels, strides, paddings):
        super().__init__()
        self.latent_dim=latent_dim
        self.im_size= im_size
        self.im_channels= im_channels
        
        activation=nn.RelU()
        layers_dim=[self.latent_dim]+conv_channels+[self.im_channels]
        self.layers=nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(layers_dim[i], layers_dim[i+1],
                                kernel_size=kernels[i],
                                stride=strides[i],
                                padding=paddings[i],
                                bias=False)
                nn.BatchNorm2d(layers_dim[i+1]) if i!=len(layers_dim)-2 else nn.Identity(),
                activation if i!=len(layers_dim)-2 else nn.Tanh()
            )
            for i in range(len(layers_dim)-1)
        ])
        
        def forward(self, z):
            batch_size=z.shape[0]
            out=z.reshape(-1,self.latent_dim,1,1)
            for layer in self.layers:
                out=layer(out)
                
            out=out.view(batch_size,self.im_channels,self.im_size,self.im_size)
            return out
        
class Discriminator(nn.Module):
    def __init__(self, im_size, im_channels, conv_channels, kernels, strides, paddings):
        super().__init__()
        self.im_size=im_size
        self.im_channels=im_channels
        
        activation=nn.LeakyReLU()
        layers_dim=[self.im_channels]+conv_channels+[1]
        self.layers=nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(layers_dim[i], layers_dim[i+1],
                          kernel_size=kernels[i],
                          stride=strides[i],
                          padding=paddings[i],
                          bias=False),
                nn.BatchNorm2d(layers_dim[i+1]) if i!=len(layers_dim)-2 and i!=0 else nn.Identity(),
                activation if i!=len(layers_dim)-2 else nn.Identity()
            )
            for i in range(len(layers_dim)-1)
        ])
        
    def forward(self, x):
        out=x
        for layer in self.layers:
            out=layer(out)
        
        return out.reshape(x.size(0))
    
if __name__ == '__main__':
    x = torch.randn((2, 100))
    out = Generator()(x)
    print(out.shape)
    prob = Discriminator()(out)




