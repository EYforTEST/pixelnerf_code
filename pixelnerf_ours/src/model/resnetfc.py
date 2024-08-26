from torch import nn
import torch

#  import torch_scatter
import torch.autograd.profiler as profiler
import util

import math
import torch.nn.functional as F

from .code import PositionalEncoding

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

class Siren(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        w0 = 1.,
        c = 6.,
        is_first = False,
        use_bias = True,
        activation = None,
        dropout = 0.
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation
        self.dropout = nn.Dropout(dropout)

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out =  F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        out = self.dropout(out)
        
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))

        out = avg_out + max_out

        return (out)
        # return self.sigmoid(out)
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        return (x)
        # return self.sigmoid(x)

# Resnet Blocks
class ResnetBlockFC(nn.Module):
    """
    Fully connected ResNet Block class.
    Taken from DVR code.
    :param size_in (int): input dimension
    :param size_out (int): output dimension
    :param size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out = None, size_h=None, beta=0.0):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)

        # Init
        nn.init.constant_(self.fc_0.bias, 0.0)
        nn.init.kaiming_normal_(self.fc_0.weight, a=0, mode="fan_in")

        nn.init.constant_(self.fc_1.bias, 0.0)
        nn.init.zeros_(self.fc_1.weight)

        self.activation = nn.ReLU() # Sine(w0 = 1.0) 

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out) # , bias=False
            
            nn.init.constant_(self.shortcut.bias, 0.0)
            nn.init.kaiming_normal_(self.shortcut.weight, a=0, mode="fan_in")

    def forward(self, x):
        with profiler.record_function("resblock"):
            net = self.fc_0(self.activation(x))
            dx = self.fc_1(self.activation(net))

            if self.shortcut is not None:
                x_s = self.shortcut(x)
            else:
                x_s = x
            return x_s + dx

class ResnetFC(nn.Module):
    def __init__(
        self, d_in, d_out=4, n_blocks=5, d_latent=0, d_hidden=128, beta=0.0, combine_layer=1000, combine_type="average", use_spade=False):
        """
        :param d_in input size
        :param d_out output size
        :param n_blocks number of Resnet blocks
        :param d_latent latent size, added in each resnet block (0 = disable)
        :param d_hidden hiddent dimension throughout network
        :param beta softplus beta, 100 is reasonable; if <=0 uses ReLU activations instead
        """
        super().__init__()
        
        self.d_out = d_out

        self.d_latent = d_latent
        self.d_hidden = d_hidden 

        self.combine_type = combine_type
        self.use_spade = use_spade      

        self.siren_module = False # if True, class ResnetBlockFC Set (activation = Sine())

        self.CH, self.NS = 512, 3

        self.coarse_samp_n = 64
        self.fine_samp_n = 192

        ############################################# init_block for Ours basic version with positional encoding  

        if self.siren_module:
            self.lin_in_xyz = Siren(dim_in = 3, dim_out = d_hidden, w0 = 1.0, use_bias = True, is_first = 0, dropout = 0.)
            self.lin_in_view = Siren(dim_in = 3, dim_out = d_hidden, w0 = 1.0, use_bias = True, is_first = 0, dropout = 0.)

            self.lin_in_depth = Siren(dim_in = d_latent, dim_out = d_hidden, w0 = 1.0, use_bias = True, is_first = 0, dropout = 0.)

            self.lin_z = nn.ModuleList([Siren(dim_in = d_latent, dim_out = d_hidden, w0 = 1.0, use_bias = True, is_first = 0, dropout = 0.) for _ in range(4)])
        else:
            self.lin_in_xyz = nn.Linear(3*10*2 + 3, d_hidden)
            self.lin_in_view = nn.Linear(3*10*2 + 3, d_hidden)

            self.lin_in_depth = nn.Linear(d_latent, d_hidden)

            self.lin_z = nn.ModuleList([nn.Linear(d_latent, d_hidden) for _ in range(4)])

        self.blocks = nn.ModuleList([ResnetBlockFC(d_hidden, beta=beta) for _ in range(8)])

        self.activation = nn.ReLU()

        self.lin_out = nn.Linear(d_hidden, d_out)
        nn.init.constant_(self.lin_out.bias, 0.0)
        nn.init.kaiming_normal_(self.lin_out.weight, a=0, mode="fan_in")

        self.positional_encoding = PositionalEncoding(num_freqs=10, d_in=3)

        '''
        ############################################# init_block for Ours split version with positional encoding

        self.lin_in_xyz_color = nn.Linear(3*10*2 + 3, d_hidden) # octave 10 positional encoding
        self.lin_in_xyz_sigma = nn.Linear(3*3*2 + 3, d_hidden) # octave 3 positional encoding

        self.lin_in_view = nn.Linear(3, d_hidden) # not use positional encoding

        self.lin_in_z = nn.ModuleList([nn.Linear(d_latent, d_hidden) for _ in range(4)])
        self.lin_in_d = nn.ModuleList([nn.Linear(d_latent, d_hidden) for _ in range(4)])

        self.blocks_sigma = nn.ModuleList([ResnetBlockFC(d_hidden, beta=beta) for _ in range(7)])
        self.blocks_color = nn.ModuleList([ResnetBlockFC(d_hidden, beta=beta) for _ in range(7)])  
        
        self.activation = nn.ReLU()
        
        self.lin_out_sigma = nn.Linear(d_hidden, 1)
        nn.init.constant_(self.lin_out_sigma.bias, 0.0)
        nn.init.kaiming_normal_(self.lin_out_sigma.weight, a=0, mode="fan_in")

        self.lin_out_color = nn.Linear(d_hidden, 3)
        nn.init.constant_(self.lin_out_color.bias, 0.0)
        nn.init.kaiming_normal_(self.lin_out_color.weight, a=0, mode="fan_in")

        self.positional_encoding_xyz_color = PositionalEncoding(num_freqs=10, d_in=3)
        self.positional_encoding_xyz_sigma = PositionalEncoding(num_freqs=3, d_in=3)
        '''

    def forward(self, if_coarse, zx, combine_inner_dims=(1,), combine_index=None, dim_size=None):
        with profiler.record_function("resnetfc_infer"):           
            z = zx[..., : self.d_latent] # z: torch.Size([4800*NS, 512]) 
            z_ = z.reshape(self.NS,-1,self.CH) # z: torch.Size([3*5120, 512]) | z_: torch.Size([3, 5120, 512])

            channel = ChannelAttention(in_planes=self.CH).to(zx.device)
            spatial = SpatialAttention().to(zx.device)

            if if_coarse:
                tot_len = z_.shape[1]//self.coarse_samp_n

                z_ = z_.reshape(self.NS, tot_len, self.coarse_samp_n, self.CH) # torch.Size([V 3, R 2, S 64, C 512])

                for ns in range(self.NS):
                    now_z_ = z_[ns] # torch.Size([R 2, S 64, C 512])
                    now_z_ = now_z_.unsqueeze(-1) # torch.Size([R 2, S 64, C 512, X 1])

                    ch_in = now_z_.permute(0, 2, 1, 3) # torch.Size([R 2, C 512, # S 64, X 1])
                    sp_in = now_z_ # torch.Size([R 2, # S 64, C 512, X 1])

                    ch_out  = channel(ch_in.float()).squeeze(-1).squeeze(-1).unsqueeze(1).expand(-1, self.coarse_samp_n, self.CH)
                    sp_out  = spatial(sp_in.float()).squeeze(-1).expand(-1, self.coarse_samp_n, self.CH) # .squeeze(1)
                        
                    sp_ch_out = ch_out + sp_out # torch.Size([R, 64, 512])
      
                    if ns == 0: fin_out = sp_ch_out.reshape(self.coarse_samp_n*tot_len, self.CH) # torch.Size([R*64, 512])
                    else: fin_out = torch.cat((fin_out, sp_ch_out.reshape(self.coarse_samp_n*tot_len, self.CH)), axis = 0) # torch.Size([ns*R*64, 512])

                our_z = fin_out + z
            else:
                our_z = 2*z

            ########################################################################################## model training for Ours basic version with positional encoding 
            
            d = zx[..., self.d_latent : self.d_latent*2] # d: torch.Size([4800*NS, 512]) 
            x = zx[..., self.d_latent*2 :] # x: torch.Size([4800*NS, 3+3]) 

            # if self.siren_module = False
            view = self.lin_in_view(self.positional_encoding(x[:,3:]))
            x = self.lin_in_xyz(self.positional_encoding(x[:,:3]))

            # if self.siren_module = True
            # view = self.lin_in_view((x[:,3:]))
            # x = self.lin_in_xyz((x[:,:3]))

            for blkid in range(8):
                if blkid in [0]: x = x + self.lin_in_depth(d)

                if blkid in [1,2,3]: x = x + self.lin_z[blkid-1](our_z)
                if blkid in [4]: x = x + self.lin_z[-1](z)

                if blkid in [5]: x = x + view
                if blkid in [6]: x = util.combine_interleaved(x, combine_inner_dims, self.combine_type)

                x = self.blocks[blkid](x) # torch.Size([1, 4800, 512])

            out = self.lin_out(self.activation(x)) # torch.Size([1, 4800, 4])
            
            '''
            ########################################################################################## model training for Ours Split version with positional encoding 
            
            d = zx[..., self.d_latent : self.d_latent*2] # d: torch.Size([4800*NS, 512])
            x = zx[..., self.d_latent*2:] # x: torch.Size([4800*NS, 3 (xyz) + 3 (view)])

            view = self.lin_in_view((x[:,3:]))
            xyz_color = self.lin_in_xyz_color(self.positional_encoding_xyz_color(x[:,:3]))
            xyz_sigma = self.lin_in_xyz_sigma(self.positional_encoding_xyz_sigma(x[:,:3]))

            sigma, color = xyz_sigma, xyz_color
            
            # newnet3
            for blkid in range(7): # self.n_blocks
                if blkid == 0:
                    sigma = sigma + self.lin_in_d[blkid](d + 0.5*our_z)
                    sigma = self.blocks_sigma[blkid](sigma) 

                    color = color + self.lin_in_z[blkid](our_z)
                    color = self.blocks_color[blkid](color)

                if blkid in [1,2,3]: 
                    sigma = sigma + self.lin_in_d[blkid](d + 0.5*our_z)
                    sigma = self.blocks_sigma[blkid](sigma) 

                    color = color + self.lin_in_z[blkid](our_z) + sigma
                    color = self.blocks_color[blkid](color)
                
                if blkid == 4: # view
                    sigma = sigma + view
                    sigma = self.blocks_sigma[blkid](sigma)

                    color = color + view
                    color = self.blocks_color[blkid](color)

                if blkid == 5:
                    sigma = util.combine_interleaved(sigma, combine_inner_dims, self.combine_type)
                    sigma = self.blocks_sigma[blkid](sigma) 

                    color = util.combine_interleaved(color, combine_inner_dims, self.combine_type)
                    color = self.blocks_color[blkid](color)
                
                if blkid == 6:
                    sigma = self.blocks_sigma[blkid](sigma) 
                    color = self.blocks_color[blkid](color)

            sigma = self.lin_out_sigma(self.activation(sigma)) # torch.Size([1, 4800, 1])
            color = self.lin_out_color(self.activation(color)) # torch.Size([1, 4800, 3])

            out = torch.cat((color, sigma), axis = -1)
            '''

            return out

    @classmethod
    def from_conf(cls, conf, d_in, **kwargs):
        # PyHocon construction
        return cls(
            d_in,
            n_blocks=conf.get_int("n_blocks", 5),
            d_hidden=conf.get_int("d_hidden", 128),
            beta=conf.get_float("beta", 0.0),
            combine_layer=conf.get_int("combine_layer", 1000),
            combine_type=conf.get_string("combine_type", "average"),  # average | max
            use_spade=conf.get_bool("use_spade", False),
            **kwargs
        )
