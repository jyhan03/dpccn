# -*- coding: utf-8 -*-
"""
Created on Sun May 23 14:42:21 2021

@author: Jyhan
"""

import torch as th
import torch.nn as nn

from typing import Tuple, List
from memonger import SublinearSequential

from libs.conv_stft import ConvSTFT, ConviSTFT 

def param(nnet, Mb=True):
    """
    Return number parameters(not bytes) in nnet
    """
    neles = sum([param.nelement() for param in nnet.parameters()])
    
    return neles / 10**6 if Mb else neles


class Conv2dBlock(nn.Module):
    def __init__(self, 
                 in_dims: int = 16,
                 out_dims: int = 32,
                 kernel_size: Tuple[int] = (3, 3),
                 stride: Tuple[int] = (1, 1),
                 padding: Tuple[int] = (1, 1)) -> None:
        super(Conv2dBlock, self).__init__() 
        self.conv2d = nn.Conv2d(in_dims, out_dims, kernel_size, stride, padding)     
        self.elu = nn.ELU()
        self.norm = nn.InstanceNorm2d(out_dims)
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.conv2d(x)
        x = self.elu(x)
        
        return self.norm(x)


class ConvTrans2dBlock(nn.Module):
    def __init__(self, 
                 in_dims: int = 32,
                 out_dims: int = 16,
                 kernel_size: Tuple[int] = (3, 3),
                 stride: Tuple[int] = (1, 2),
                 padding: Tuple[int] = (1, 0),
                 output_padding: Tuple[int] = (0, 0)) -> None:
        super(ConvTrans2dBlock, self).__init__() 
        self.convtrans2d = nn.ConvTranspose2d(in_dims, out_dims, kernel_size, stride, padding, output_padding)     
        self.elu = nn.ELU()
        self.norm = nn.InstanceNorm2d(out_dims)
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.convtrans2d(x)
        x = self.elu(x)
        
        return self.norm(x)
    
    
class DenseBlock(nn.Module):
    def __init__(self, in_dims, out_dims, mode = "enc", **kargs):
        super(DenseBlock, self).__init__()
        if mode not in ["enc", "dec"]:
            raise RuntimeError("The mode option must be 'enc' or 'dec'!")
            
        n = 1 if mode == "enc" else 2
        self.conv1 = Conv2dBlock(in_dims=in_dims*n, out_dims=in_dims, **kargs)
        self.conv2 = Conv2dBlock(in_dims=in_dims*(n+1), out_dims=in_dims, **kargs)
        self.conv3 = Conv2dBlock(in_dims=in_dims*(n+2), out_dims=in_dims, **kargs)
        self.conv4 = Conv2dBlock(in_dims=in_dims*(n+3), out_dims=in_dims, **kargs)
        self.conv5 = Conv2dBlock(in_dims=in_dims*(n+4), out_dims=out_dims, **kargs)
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        y1 = self.conv1(x)
        y2 = self.conv2(th.cat([x, y1], 1))
        y3 = self.conv3(th.cat([x, y1, y2], 1))
        y4 = self.conv4(th.cat([x, y1, y2, y3], 1))
        y5 = self.conv5(th.cat([x, y1, y2, y3, y4], 1))
        
        return y5
    
        
class TCNBlock(nn.Module):
    """
    TCN block:
        IN - ELU - Conv1D - IN - ELU - Conv1D
    """

    def __init__(self,
                 in_dims: int = 384,
                 out_dims: int = 384,
                 kernel_size: int = 3,
                 stride: int = 1,
                 paddings: int = 1,
                 dilation: int = 1,
                 causal: bool = False) -> None:
        super(TCNBlock, self).__init__()
        self.norm1 = nn.InstanceNorm1d(in_dims)
        self.elu1 = nn.ELU()
        dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        # dilated conv
        self.dconv1 = nn.Conv1d(
            in_dims,
            out_dims,
            kernel_size,
            padding=dconv_pad,
            dilation=dilation,
            groups=in_dims,
            bias=True)
        
        self.norm2 = nn.InstanceNorm1d(in_dims)
        self.elu2 = nn.ELU()    
        self.dconv2 = nn.Conv1d(in_dims, out_dims, 1, bias=True)
        
        # different padding way
        self.causal = causal
        self.dconv_pad = dconv_pad

    def forward(self, x: th.Tensor) -> th.Tensor:
        y = self.elu1(self.norm1(x))
        y = self.dconv1(y)
        if self.causal:
            y = y[:, :, :-self.dconv_pad]
        y = self.elu2(self.norm2(y))
        y = self.dconv2(y)    
        x = x + y

        return x 
    
     
class DenseUNet(nn.Module):
    def __init__(self, 
                 win_len: int = 512,    # 32 ms
                 win_inc: int = 128,    # 8 ms
                 fft_len: int = 512,
                 win_type: str = "sqrthann",                 
                 kernel_size: Tuple[int] = (3, 3),
                 stride1: Tuple[int] = (1, 1),
                 stride2: Tuple[int] = (1, 2),
                 paddings: Tuple[int] = (1, 0),
                 output_padding: Tuple[int] = (0, 0),
                 tcn_dims: int = 384,
                 tcn_blocks: int = 10,
                 tcn_layers: int = 2,
                 causal: bool = False,     
                 pool_size: Tuple[int] = (4, 8, 16, 32),
                 num_spks: int = 2) -> None:
        super(DenseUNet, self).__init__()
        
        self.fft_len = fft_len
        self.num_spks = num_spks
        
        self.stft = ConvSTFT(win_len, win_inc, fft_len, win_type, 'complex')
        self.conv2d = nn.Conv2d(2, 16, kernel_size, stride1, paddings)                
        self.encoder = self._build_encoder(
                    kernel_size=kernel_size,
                    stride=stride2,
                    padding=paddings
                )
        self.tcn_layers = self._build_tcn_layers(
                    tcn_layers,
                    tcn_blocks,
                    in_dims=tcn_dims,
                    out_dims=tcn_dims,
                    causal=causal                         
                )
        self.decoder = self._build_decoder(
                    kernel_size=kernel_size,
                    stride=stride2,
                    padding=paddings,
                    output_padding=output_padding
                )
        self.avg_pool = self._build_avg_pool(pool_size)
        self.avg_proj = nn.Conv2d(64, 32, 1, 1)
        
        self.deconv2d = nn.ConvTranspose2d(32, 2*num_spks, kernel_size, stride1, paddings)
        self.istft = ConviSTFT(win_len, win_inc, fft_len, win_type, 'complex')
        
    def _build_encoder(self, **enc_kargs):
        """
        Build encoder layers 
        """
        encoder = nn.ModuleList()
        encoder.append(DenseBlock(16, 16, "enc"))
        for i in range(4):
            encoder.append(
                    SublinearSequential(
                            Conv2dBlock(in_dims=16 if i==0 else 32, 
                                    out_dims=32, **enc_kargs),
                            DenseBlock(32, 32, "enc")
                            )
                    )
        encoder.append(Conv2dBlock(in_dims=32, out_dims=64, **enc_kargs))
        encoder.append(Conv2dBlock(in_dims=64, out_dims=128, **enc_kargs))
        encoder.append(Conv2dBlock(in_dims=128, out_dims=384, **enc_kargs))

        return encoder
    
    def _build_decoder(self, **dec_kargs):
        """
        Build decoder layers 
        """
        decoder = nn.ModuleList()
        decoder.append(ConvTrans2dBlock(in_dims=384*2, out_dims=128, **dec_kargs))
        decoder.append(ConvTrans2dBlock(in_dims=128*2, out_dims=64, **dec_kargs))
        decoder.append(ConvTrans2dBlock(in_dims=64*2, out_dims=32, **dec_kargs))        
        for i in range(4):
            decoder.append(
                    SublinearSequential(
                            DenseBlock(32, 64, "dec"),
                            ConvTrans2dBlock(in_dims=64, 
                                             out_dims=32  if i!=3 else 16,
                                             **dec_kargs)
                            )
                    )
        decoder.append(DenseBlock(16, 32, "dec"))                            
        
        return decoder    
    
    def _build_tcn_blocks(self, tcn_blocks, **tcn_kargs):
        """
        Build TCN blocks in each repeat (layer)
        """
        blocks = [
            TCNBlock(**tcn_kargs, dilation=(2**b))
            for b in range(tcn_blocks)
        ]
        
        return SublinearSequential(*blocks)
    
    def _build_tcn_layers(self, tcn_layers, tcn_blocks, **tcn_kargs):
        """
        Build TCN layers
        """
        layers = [
            self._build_tcn_blocks(tcn_blocks, **tcn_kargs)
            for _ in range(tcn_layers)
        ]
        
        return SublinearSequential(*layers)
    
    def _build_avg_pool(self, pool_size):
        """
        Build avg pooling layers
        """
        avg_pool = nn.ModuleList()
        for sz in pool_size:
            avg_pool.append(
                    SublinearSequential(
                            nn.AvgPool2d(sz),
                            nn.Conv2d(32, 8, 1, 1)                            
                            )
                )
        
        return avg_pool
    
    def wav2spec(self, x: th.Tensor, mags: bool = False) -> th.Tensor:
        """
        convert waveform to spectrogram
        """
        assert x.dim() == 2 
        x = x / th.std(x, -1, keepdims=True)        # variance normalization
        specs = self.stft(x)
        real = specs[:,:self.fft_len//2+1]
        imag = specs[:,self.fft_len//2+1:]
        spec = th.stack([real,imag], 1)
        spec = th.einsum("hijk->hikj", spec)    # batchsize, 2, T, F        
        if mags:
            return th.sqrt(real**2+imag**2+1e-8)
        else:
            return spec    
    
    def sep(self, spec: th.Tensor) -> List[th.Tensor]:
        """
        spec: (batchsize, 2*num_spks, T, F)
        return [real, imag] or waveform for each speaker
        """
        spec = th.einsum("hijk->hikj", spec)        # (batchsize, 2*num_spks, F, T)
        spec = th.chunk(spec, self.num_spks, 1)
        B, N, F, T = spec[0].shape
        est1 = th.chunk(spec[0], 2, 1)      # [(B, 1, F, T), (B, 1, F, T)]
        est2 = th.chunk(spec[1], 2, 1)  
        est1 = th.cat(est1, 2).reshape(B, -1, T)      # B, 1, 2F, T
        est2 = th.cat(est2, 2).reshape(B, -1, T)  
        return [th.squeeze(self.istft(est1)), th.squeeze(self.istft(est2))]
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        if x.dim() == 1:
            x = th.unsqueeze(x, 0)
            
        spec = self.wav2spec(x)
        out = self.conv2d(spec)
        out_list = []
        for _, enc in enumerate(self.encoder):
            out = enc(out)
            out_list.append(out)
        B, N, T, F = out.shape
        out = self.tcn_layers(out.reshape(B, N, T*F))
        out = th.unsqueeze(out, -1)

        out_list = out_list[::-1]
        for idx, dec in enumerate(self.decoder):
            out = dec(th.cat([out_list[idx], out], 1))   
          
        # Pyramidal pooling
        B, N, T, F = out.shape
        upsample = nn.Upsample(size=(T, F), mode='bilinear')
        pool_list = []
        for avg in self.avg_pool:
            pool_list.append(upsample(avg(out)))
            
        out = th.cat([out, *pool_list], 1)
        out = self.avg_proj(out)
        out = self.deconv2d(out)
        
        return self.sep(out)
       
    
def test_covn2d_block():
    x = th.randn(2, 16, 257, 200)
    conv = Conv2dBlock()
    y = conv(x)
    print(y.shape)
    convtrans = ConvTrans2dBlock()
    z = convtrans(y)
    print(z.shape)
    
def test_dense_block():
    x = th.randn(2, 16, 257, 200)
    dense = DenseBlock(16, 32, "enc")
    y = dense(x)
    print(y.shape)
    
def test_tcn_block():
    x = th.randn(2, 384, 1000)
    tcn = TCNBlock(dilation=128)
    print(tcn(x).shape)


if __name__ == "__main__":
    nnet = DenseUNet()
    print(param(nnet))
    x = th.randn(2, 32000)
    est1, est2 = nnet(x)
