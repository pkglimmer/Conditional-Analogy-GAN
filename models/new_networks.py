import torch
from torch import nn
from collections import OrderedDict
from torch.nn import init
from functools import partial
import torch.nn.functional as F

# UnetG: CAGAN generator I implemented 
# UnetDilate: StackGAN (code borrowed from https://github.com/maktu6/CAGAN_v2_pytorch/blob/master/utils/network.py)
        
class Conv2dSame(nn.Module):
    """ 
    Conv2d with 'same padding', equivalent to 'Conv2d(...,padding='same')' in keras
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding_layer=nn.ZeroPad2d, dilation=1, bias=True):
        super().__init__()
        
        kernel_h, kernel_w = kernel_size
        
        k = dilation*(kernel_w-1)-stride+2
        pr = k // 2
        pl = pr - 1 if k % 2 == 0 else pr
        
        k = dilation*(kernel_h-1)-stride+1+1
        pb = k // 2
        pt = pb - 1 if k % 2 == 0 else pb
        self.pad_same = padding_layer((pl, pr, pb, pt))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride, dilation=dilation, bias=bias)

    def forward(self, x):
        x = self.pad_same(x)
        x = self.conv(x)
        return x

class Unet_G(nn.Module):
    def __init__(self, input_nc=9, output_nc=4, img_size=128, ngf=64):
        super(Unet_G, self).__init__()
        _conv = {'kernel_size': (4,3), 'stride': 2, 'bias': True}
        self.InstanceNorm = partial(nn.InstanceNorm2d, affine=True)
        # self.relu = partial(nn.ReLU, True)
        # self.leaky_relu = partial(nn.LeakyReLU, 0.2, True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.pool2 = nn.AvgPool2d(2)
        self.pool4 = nn.AvgPool2d(4)
        self.pool8 = nn.AvgPool2d(8)
        self.pool16 = nn.AvgPool2d(16)

        # down
        self.conv1 = Conv2dSame(9, 64, **_conv)
        self.conv2 = Conv2dSame(64+6, 128, **_conv)
        self.norm2 = nn.InstanceNorm2d(128)
        self.conv3 = Conv2dSame(128+6, 256, **_conv)
        self.norm3 = nn.InstanceNorm2d(256)
        self.conv4 = Conv2dSame(256+6, 512, **_conv)
        self.norm4 = nn.InstanceNorm2d(512)

        #up
        self.upconv5 = nn.ConvTranspose2d(512+6,256,kernel_size=4,stride=2,padding=1, bias=True)
        self.norm5 = nn.InstanceNorm2d(256)
        self.upconv6 = nn.ConvTranspose2d(512+6,128,kernel_size=4, stride=2,padding=1, bias=True)
        self.pad6 = nn.ReflectionPad2d((1,1,0,0))
        self.norm6 = nn.InstanceNorm2d(128)
        self.upconv7 = nn.ConvTranspose2d(256+6,64,kernel_size=4, stride=2,padding=1, bias=True)
        self.pad7 = nn.ReflectionPad2d((1,1,0,0))
        self.norm7 = nn.InstanceNorm2d(64)
        self.upconv8 = nn.ConvTranspose2d(64+6,32,kernel_size=4, stride=2,padding=1, bias=True)
        self.norm8 = nn.InstanceNorm2d(32)
        self.conv9 = Conv2dSame(32, 4, kernel_size=(4,3), stride=1, bias=True)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input):
        xi = input[:,0:3,:,:]
        yi = input[:,3:6,:,:]
        xi_yi = torch.cat([xi, yi], dim=1)
        xi_yi_sz64 = self.pool2(xi_yi)
        xi_yi_sz32 = self.pool4(xi_yi)
        xi_yi_sz16 = self.pool8(xi_yi)
        xi_yi_sz8 = self.pool16(xi_yi)

        # Encoder
        l1 = self.leaky_relu(self.conv1(input))
        l1 = torch.cat([l1, xi_yi_sz64], dim=1)
        l2 = self.norm2(self.conv2(l1))
        l3 = self.leaky_relu(l2)
        l3 = torch.cat([l3, xi_yi_sz32], dim=1)
        l3 = self.norm3(self.conv3(l3))
        l4 = self.leaky_relu(l3)
        l4 = torch.cat([l4, xi_yi_sz16], dim=1)
        l4 = self.leaky_relu(self.norm4(self.conv4(l4)))
        l4 = torch.cat([l4, xi_yi_sz8], dim=1)
        # Decoder
        l5 = self.upconv5(l4) 
        l5 = self.norm5(l5)
        l5 = torch.cat([l5,l3], dim=1)
        l5 = self.relu(l5) # [8, 512, 16, 12]
        l5 = torch.cat([l5,xi_yi_sz16], dim=1) # [8, 6, 16, 12]
        l6 = self.upconv6(l5)
        l6 = self.norm6(l6)        
        l6 = torch.cat([l6,l2], dim=1)
        l6 = self.relu(l6) # [8, 256, 32, 24]
        l6 = torch.cat([l6,xi_yi_sz32], dim=1) # [8, 256+6, 32, 24]
        l7 = self.upconv7(l6) # to 64 channels
        l7 = self.relu(self.norm7(l7)) # [8, 64, 64, 46]
        l8 = torch.cat([l7,xi_yi_sz64], dim=1) # [8, 70, 64, 46]
        l8 = self.upconv8(self.relu(l8)) # [8, 32, 128, 92]
        # l8 = l8[:,:,:-1,:-1]
        l8 = self.norm8(l8)
        l9 = self.conv9(l8)

        alpha = l9[:,0:1,:,:] # [8, 1, 128, 92]
        xij = l9[:,1:,:,:]
        alpha = self.sigmoid(alpha)
        xij = self.tanh(xij)
        out = torch.cat([alpha, xij], dim=1)
        return out

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.
    Args:
        stddev: float, standard deviation of the noise distribution.
    """

    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev
        self.noise = torch.tensor(0.0)

    def forward(self, x):
        if self.stddev != 0:
            sampled_noise = self.noise.repeat(*x.size()).normal_(std=self.stddev).to(x.device)
            x = x + sampled_noise
        return x 

class Basic_D(nn.Module):
    """DCGAN_D(nc, ndf, max_layers=3)
       nc: channels
       ndf: filters of the first layer
       max_layers: max hidden layers
    """
    def __init__(self, nc_in, ndf, max_layers=3, use_sigmoid=True):
        super(Basic_D, self).__init__()
        basic_d = OrderedDict([('noise', GaussianNoise(0.05)),
                               ('conv0', Conv2dSame(nc_in, ndf, (4,4), 2)),
                               ('Lrelu0', nn.LeakyReLU(0.2, True))])
        in_feat = ndf
        for i in range(1, max_layers):
            out_feat = ndf * min(2**i, 8)
            basic_d.update([
                    ('conv%d'%i, Conv2dSame(in_feat, out_feat, (4,4), 2, bias=False)),
                    ('bn%d'%(i-1), nn.InstanceNorm2d(out_feat, affine=True)),#BN or IN(original)
                    ('Lrelu%d'%i, nn.LeakyReLU(0.2, True))])
            in_feat = ndf * min(2**i, 8)
        
        out_feat = ndf * min(2**max_layers, 8)
        basic_d.update([
                        ('pad1', nn.ZeroPad2d(1)),
                        ('conv%d'%(max_layers+1), nn.Conv2d(in_feat, out_feat, 4, bias=False)),
                        ('bn%d'%(max_layers+1), nn.InstanceNorm2d(out_feat, affine=True)),
                        ('Lrelu%d'%(max_layers+1), nn.LeakyReLU(0.2, True)),
                        ('pad2', nn.ZeroPad2d(1)),
                        ('conv%d'%(max_layers+2), nn.Conv2d(out_feat, 1, 4))])
        if use_sigmoid:
            basic_d['sigmoid'] = nn.Sigmoid()
        self.D_net = nn.Sequential(basic_d)
        
    def forward(self, x):
        return self.D_net(x)


class UpscaleBlock(nn.Module):
    def __init__(self, nf_in, nf_out, up_type='Tconv'):
        super(UpscaleBlock, self).__init__()
        block = OrderedDict()
        if (up_type == 'nearest') or (up_type == 'bilinear'):
            block['upsample'] = nn.Upsample(scale_factor=2, mode='nearest')
            block['conv'] = Conv2dSame(nf_in, nf_out, (4,3), bias=False)
        elif up_type == 'Tconv':
            block['convT'] = nn.ConvTranspose2d(nf_in, nf_out, (4,3), 2,
                                                padding=1, bias=False)
            block['pad'] = nn.ReflectionPad2d((0,1,0,0))
        elif up_type == 'ps':
            block['conv'] = Conv2dSame(nf_in, nf_out*4, (4,3), bias=False)
            block['ps'] = nn.PixelShuffle(2)
        block['bn'] = nn.InstanceNorm2d(nf_out, affine=True)
        self.up_block = nn.Sequential(block)
        
    def forward(self, x):
        return self.up_block(x)

class SE_Block(nn.Module):
    def __init__(self, nc, reduction=8):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(nc, nc//reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(nc//reduction, nc, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class Res_SE_Block(nn.Module):
    def __init__(self, nc_in, nc_out=None, use_instancenorm=True):
        super(Res_SE_Block, self).__init__()
        if nc_out == None:
            nc_out = nc_in
        if use_instancenorm:
            norm_layer = partial(nn.InstanceNorm2d, affine=True)
        else:
            norm_layer = nn.BatchNorm2d
        res_block = OrderedDict([
            ('conv0', Conv2dSame(nc_in, nc_out, 
                                (4,3), bias=(not use_instancenorm))),
            ('bn0', norm_layer(nc_out)),
            ('relu0', nn.ReLU(True)),
            ('conv1', Conv2dSame(nc_out, nc_out,
                                (4,3), bias=(not use_instancenorm))),
            ('bn1', norm_layer(nc_out))
        ])
        self.res_block = nn.Sequential(res_block)
        self.se_block = SE_Block(nc_out)
        
    def forward(self, x):
        module_input = x
        x = self.res_block(x)
        x = self.se_block(x)
        return x + module_input
        
class Refiner_Network(nn.Module):
    def __init__(self, nc_in, nc_out, nc_res=32):
        super(Refiner_Network, self).__init__()
        self.conv0 = nn.Conv2d(nc_in, nc_res, 1, bias=False)
        for i in range(2):
            setattr(self, 'res_se_block%d'%i, Res_SE_Block(nc_res))
        self.conv1 = nn.Conv2d(nc_res, nc_out, 1, bias=False)
    
    def forward(self, x):
        module_input = x
        x = self.conv0(x)
        for i in range(2):
            x = getattr(self, 'res_se_block%d'%i)(x)
        x = self.conv1(x)
        return x + module_input

class Out_Branch(nn.Module):
    def __init__(self, nc_in, kernel_size=(4,3)):
        super(Out_Branch, self).__init__()
        self.conv = Conv2dSame(nc_in, 4, kernel_size, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.conv(x)
        alpha = x[:, 0:1, :, :]
        x_i_j = x[:, 1:, :, :]
        alpha = self.sigmoid(alpha)
        x_i_j = self.tanh(x_i_j)
        return torch.cat([alpha, x_i_j], 1)
        
class Unet_Dilate(nn.Module):
    def __init__(self, nc_in, nc_out, ngf=64, use_instancenorm=True, up_type='Tconv'):
        super(Unet_Dilate, self).__init__()
        if use_instancenorm:
            self.norm_layer = partial(nn.InstanceNorm2d, affine=True)
        else:
            self.norm_layer = nn.BatchNorm2d 
        relu = partial(nn.ReLU, True)
        leak_relu = partial(nn.LeakyReLU, 0.2, True)
        
        self.pool0_0 = nn.AvgPool2d(2)
        self.pool0_1 = nn.AvgPool2d(4)
        self.layer1 = nn.Sequential(
            Conv2dSame(nc_in, 64, (4,3), 2, bias=False),
            leak_relu()
        )
        self.layer2 = self.make_layer(64+6, 128, 2)
        self.Lrelu3 = leak_relu()
        self.layer3 = self.make_layer(128+6, 256, 1)
        self.Lrelu4 = leak_relu()
        self.layer4 = self.make_layer(256+6, 256, 1, 2, relu_layer=leak_relu)
        self.layer5 = self.make_layer(256+6, 256, 1, 2, relu_layer=leak_relu)
        self.layer8 = self.make_layer(256+6, 256, 1, 2, relu_layer=relu)
        self.layer9 = self.make_layer(256*2+6, 256, 1, 2)
        self.relu9 = relu()
        self.layer10 = self.make_layer(256*2+6, 128, 1)
        self.relu10 = relu()
        
        self.branch_0 = Out_Branch(128*2)
        self.refine_0 = Refiner_Network(128*2+6, 128*2+6, 64)
        self.relu11_0 = relu()
        self.upscale_0 = UpscaleBlock(128*2+6, 64, up_type=up_type)
        self.relu11_1 = relu()
        self.branch_1 = Out_Branch(64)
        self.refine_1 = Refiner_Network(64+6, 64+6, 32)
        self.relu12 = relu()
        self.upscale_1 = UpscaleBlock(64+6, 32, up_type=up_type)
        self.branch_2 = Out_Branch(32) #change kernel size from (8,6) to (4,3)
        
    def make_layer(self, nc_in, nc_out, stride, dilate=1, use_norm=True, relu_layer=None):
        layers = []
        layers.append(Conv2dSame(nc_in, nc_out, (4,3), stride, dilation=dilate, bias=False))
        if use_norm: 
            layers.append(self.norm_layer(nc_out))
        if relu_layer != None:
            layers.append(relu_layer())
        return nn.Sequential(*layers)

    def forward(self, x):
        xi =x[:, 0:3, : :]
        yj =x[:, 6:, : :]
        xi_yj = torch.cat([xi, yj], 1)
        xi_yj_sz128 = self.pool0_0(xi_yj)
        xi_yj_sz64 = self.pool0_1(xi_yj)
        x_1 = self.layer1(x)
        x_1 = torch.cat([x_1, xi_yj_sz128], 1)
        x_2 = self.layer2(x_1)
        x_3 = self.Lrelu3(x_2)
        x_3 = torch.cat([x_3, xi_yj_sz64], 1)
        x_3 = self.layer3(x_3)
        x_4 = self.Lrelu4(x_3)
        x_4 = torch.cat([x_4, xi_yj_sz64], 1)
        x_4 = self.layer4(x_4)
        x_4 = torch.cat([x_4, xi_yj_sz64], 1)
        
        x_5 = self.layer5(x_4)
        x_5 = torch.cat([x_5, xi_yj_sz64], 1)
        x_8 = self.layer8(x_5)
        x_8 = torch.cat([x_8, x_4], 1)
        x_9 = self.layer9(x_8)
        x_9 = torch.cat([x_9, x_3], 1)
        x_9 = self.relu9(x_9)
        x_9 = torch.cat([x_9, xi_yj_sz64], 1)
        x_10 = self.layer10(x_9)
        x_10 = torch.cat([x_10, x_2], 1)
        x_10 = self.relu10(x_10)
        
        out_0 = self.branch_0(x_10)
        x_10 = torch.cat([x_10, xi_yj_sz64], 1)
        x_11 = self.refine_0(x_10)
        x_11 = self.relu11_0(x_11)
        x_11 = self.upscale_0(x_11)
        x_11 = self.relu11_1(x_11)
        
        out_1 = self.branch_1(x_11)
        x_12 = torch.cat([x_11, xi_yj_sz128], 1)
        x_12 = self.refine_1(x_12)
        x_12 = self.relu12(x_12)
        x_12 = self.upscale_1(x_12)
        out_2 = self.branch_2(x_12)
        # return [out_0, out_1, out_2]
        return out_2


if __name__ == '__main__':
    # check the model: python -m utils.network
    ngf = 64
    ndf = 64
    nc_G_inp = 9 
    nc_G_out = 4 
    nc_D_inp = 6 
    
    D_net = Basic_D(nc_D_inp, ndf)
    netGA = Unet_Dilate(nc_G_inp, nc_G_out, ngf)
    netGG = Unet_G(nc_G_inp, nc_G_out, 128, ngf)

    netGA.cuda()
    D_net.cuda()
    netGG.cuda()
    batch = torch.randn((8, 9, 256, 192))
    batch = batch.cuda()
    batch128 = torch.randn((8,9,128,92))
    batch128 = batch128.cuda()

    y = netGA(batch)
    y = D_net(batch[:,0:6,:,:])
    y1 = netGG(batch128)

    num_params = 0
    for param in netGA.parameters():
        num_params += param.numel()

    print('netG:5867800', num_params)

    num_params = 0
    for param in D_net.parameters():
        num_params += param.numel()

    print('netD:2768705',num_params)
    import pdb; pdb.set_trace()
