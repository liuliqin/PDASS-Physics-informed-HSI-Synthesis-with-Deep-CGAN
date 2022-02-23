import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math


class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()

        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


class generator(nn.Module):
    # initializers
    def __init__(self, ResBlock,d=64,input_nc=3,output_nc=3,channel=48):
        super(generator, self).__init__()
        self.inchannel=d
        # Unet encoder
        self.conv1 = nn.Conv2d(input_nc,d,3,1,1) #128*128
        self.layer1 = self.make_layer(ResBlock, d, 2, stride=2) #64*64
        self.layer1_bn = nn.BatchNorm2d(d)
        self.layer2 = self.make_layer(ResBlock, d * 2, 2, stride=2)#32*32
        self.layer2_bn = nn.BatchNorm2d(2*d)
        self.layer3 = self.make_layer(ResBlock, d * 4, 2, stride=2)#16*16
        self.layer3_bn = nn.BatchNorm2d(4 * d)
        self.layer4 = self.make_layer(ResBlock, d * 8, 2, stride=2)#8*8
        self.layer4_bn = nn.BatchNorm2d(8 * d)
        self.layer5 = self.make_layer(ResBlock, d * 8, 2, stride=2)#4*4
        self.layer5_bn = nn.BatchNorm2d(8 * d)
        self.layer6 = self.make_layer(ResBlock, d * 8, 2, stride=2)#2*2

        c=torch.tensor([3.3425517,6.0359445,9.5913372,12.531060,13.093282,14.879770,
                      16.380230,16.440357,15.726861,15.352000,14.562735,14.914811,
                      14.661460,14.544858,14.140451,13.684365,12.861519,12.512197,
                      11.625054,11.056995,10.420329,9.8573799,8.4956789,8.4372854,
                      8.2261105,7.9227495,8.3297834,6.8664875,7.0682282,7.5394888,
                      7.3041086,6.7612209,6.4260387,6.6136274,6.3207126,6.2090621,
                      6.2187519,5.2961211,4.7726207,3.8006046,2.6732323,3.1805446,
                      4.3476920,5.0446596,5.2692027,5.3231392,5.3550949,5.3823729])

        self.bright= nn.Parameter(torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(c,0),2),3).cuda(),requires_grad=True)


        # Unet decoder
        self.deconv6 = nn.Conv2d(d * 8, d * 8, 3, 1, 1)#4*4
        self.deconv6_bn = nn.BatchNorm2d(d * 8)
        self.deconv5 = nn.Conv2d(d * 8, d * 8, 3, 1, 1)#8*8
        self.deconv5_bn = nn.BatchNorm2d(d * 8)
        self.deconv4 = nn.Conv2d(d * 8, d * 4, 3, 1, 1)#16*16
        self.deconv4_bn = nn.BatchNorm2d(d * 4)
        self.deconv3 = nn.Conv2d(d * 4, d * 2, 3, 1, 1)#32*32
        self.deconv3_bn = nn.BatchNorm2d(d * 2)
        self.deconv2 = nn.Conv2d(d * 2, d, 3, 1, 1)#64*64
        self.deconv2_bn = nn.BatchNorm2d(d)
        self.deconv1 = nn.Conv2d(d, d, 3, 1, 1)  # 128*128
        self.deconv1_bn = nn.BatchNorm2d(d)
        self.conv2 = nn.Conv2d(d, output_nc, kernel_size=3, stride=1, padding=1)# 128*128*output_c




    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        encode1 = self.conv1(input)#128*128
        encode2 = self.layer1_bn(self.layer1(F.leaky_relu(encode1, 0.2)))#64*64
        encode3 = self.layer2_bn(self.layer2(F.leaky_relu(encode2, 0.2)))#32*32
        encode4 = self.layer3_bn(self.layer3(F.leaky_relu(encode3, 0.2)))#16*16
        encode5 = self.layer4_bn(self.layer4(F.leaky_relu(encode4, 0.2)))#8*8
        encode6 = self.layer5_bn(self.layer5(F.leaky_relu(encode5, 0.2)))#4*4
        encode7 = self.layer6(F.leaky_relu(encode6, 0.2))#2*2

        up7=F.interpolate(F.leaky_relu(encode7, 0.2),scale_factor=2,mode='bilinear')#4
        decode6 = F.leaky_relu(self.deconv6_bn(self.deconv6(up7)),0.2)
        up6=F.interpolate(encode6 + decode6,scale_factor=2,mode='bilinear')#8
        decode5 = F.leaky_relu(self.deconv5_bn(self.deconv5(up6)),0.2)
        up5 = F.interpolate(encode5 + decode5, scale_factor=2, mode='bilinear')#16
        decode4 = F.leaky_relu(self.deconv4_bn(self.deconv4(up5)),0.2)
        up4 = F.interpolate(encode4 + decode4, scale_factor=2, mode='bilinear')#32
        decode3 = F.leaky_relu(self.deconv3_bn(self.deconv3(up4)), 0.2)
        up3 = F.interpolate(encode3 + decode3, scale_factor=2, mode='bilinear')#64
        decode2= F.leaky_relu(self.deconv2_bn(self.deconv2(up3)), 0.2)
        up2 = F.interpolate(encode2 + decode2, scale_factor=2, mode='bilinear')#128
        decode1 = F.leaky_relu(self.deconv1_bn(self.deconv1(up2)), 0.2)
        pos = self.conv2(F.leaky_relu(encode1 + decode1, 0.2))

        # abun = F.softmax(pos)

        return pos,self.bright

class Reslearner(nn.Module):
    def __init__(self,d=64,input_nc=3):
        super(Reslearner,self).__init__()
        self.conv1=nn.Conv2d(input_nc,d,kernel_size=3,stride=1,padding=1)
        self.res1 = nn.Sequential(
            nn.Conv2d(d, d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d)
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(d, 2*d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2*d),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * d, 2 * d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 * d),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * d, d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d)
        )
        self.res3 = nn.Sequential(
            nn.Conv2d(d, 4 * d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4 * d),
            nn.ReLU(inplace=True),
            nn.Conv2d(4* d, 4 * d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4 * d),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * d, d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d)
        )
        self.conv2 = nn.Conv2d(d,input_nc,kernel_size=3,stride=1,padding=1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    def forward(self,input):
        c1=F.leaky_relu(self.conv1(input))
        r1=self.res1(c1)
        o1=F.leaky_relu(r1+c1)
        r2=self.res2(o1)
        o2=F.leaky_relu(r2+o1)
        r3 = self.res2(o2)
        o3 = F.leaky_relu(r3 + o2)
        res=F.leaky_relu(self.conv2(o3))
        out=input+res
        return out

class con_discriminator(nn.Module):
    # initializers
    def __init__(self, d=64, input_nc=6):
        super(con_discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(input_nc, d, 3, 1, 1)#6 channels, 256*256*d
        self.conv1_2 = nn.Conv2d(d, d * 2, 3, 1, 1)  # 6 channels, 256*256*2d
        self.conv2_1 = nn.Conv2d(d * 2, d * 2, 4, 2, 1)#128*128*2d
        self.conv2_2 = nn.Conv2d(d * 2, d * 4, 3, 1, 1)#128*128*4d
        self.conv2_bn = nn.BatchNorm2d(d * 4)
        self.conv3_1 = nn.Conv2d(d * 4, d * 4, 4, 2, 1)#64*64*4d
        self.conv3_2 = nn.Conv2d(d * 4, d * 8, 3, 1, 1)#64*64*8d
        self.conv3_bn = nn.BatchNorm2d(d * 8)
        self.conv4 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)# 32*32*8d
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 1)
        # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    # forward method
    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = F.leaky_relu(self.conv1_2(self.conv1_1(x)), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2_2(self.conv2_1(x))), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3_2(self.conv3_1(x))), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))

        return x

class spe_discriminator(nn.Module):
    def __init__(self,input_nc=150,inter=64,d=128):
        super(spe_discriminator, self).__init__()
        self.inter=inter
        self.fc_1=nn.Linear(input_nc,d)
        self.fc_2=nn.Linear(d,2 * d)
        self.fc_3 = nn.Linear(2 * d, 4 * d)
        self.fc_4=nn.Linear(4 * d,8 * d)
        self.fc_5 = nn.Linear(8 * d, 4 * d)
        self.fc_6=nn.Linear(4 * d,1)
    def forward(self,input_real,result):
        location_h = random.randrange(self.inter)
        location_w = random.randrange(self.inter)
        numbers=int(math.ceil(input_real.shape[2]/(self.inter)))
        real=torch.zeros(numbers,numbers,input_real.shape[0]).cuda()
        pre =torch.zeros(numbers,numbers,result.shape[0]).cuda()
        for h in range(numbers):
            for w in range(numbers):
                loc_h=location_h+self.inter*h
                loc_w=location_w+self.inter*w
                real_spectral=input_real[:,:,loc_h,loc_w]
                pre_spectral=result[:,:,loc_h,loc_w]
                spe2=F.leaky_relu(self.fc_2(F.leaky_relu((self.fc_1(real_spectral)))))
                spe3=F.leaky_relu(self.fc_3(spe2))
                spe4 = F.leaky_relu(self.fc_4(spe3))
                spe5 = F.leaky_relu(self.fc_5(spe4))
                real[h, w,:] = torch.sigmoid(F.leaky_relu(self.fc_6(spe5))).squeeze()
                pre2=F.leaky_relu(self.fc_2(F.leaky_relu((self.fc_1(pre_spectral)))))
                pre3=F.leaky_relu(self.fc_3(pre2))
                pre4 = F.leaky_relu(self.fc_4(pre3))
                pre5 = F.leaky_relu(self.fc_5(pre4))
                pre[h, w,:] = torch.sigmoid(F.leaky_relu(self.fc_6(pre5))).squeeze()
        return real.permute(2,0,1),pre.permute(2,0,1)
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()