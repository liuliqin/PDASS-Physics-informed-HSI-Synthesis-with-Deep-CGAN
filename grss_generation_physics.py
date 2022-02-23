import os,time,argparse, network
import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from datasets import ImageDataset
from util import Logger
import h5py
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='grss_dfc_2018',  help='')
parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--lib_path', required=False, default='spe_grss_345.mat', help='spectral lib path')
parser.add_argument('--ngf', type=int, default=128)
parser.add_argument('--ndf', type=int, default=128)
parser.add_argument('--crop_size', type=int, default=128, help='crop size')
parser.add_argument('--start_epoch', type=int, default=1400, help='number of start training epoch')
parser.add_argument('--lrd_start_epoch', type=int, default=400, help='number of start lr decay epoch')
parser.add_argument('--train_epoch', type=int, default=1801, help='number of train epochs')
parser.add_argument('--D_start_epoch', type=int, default=0, help='start D epoch')
parser.add_argument('--G_iter', type=int, default=3, help='G iters peer D iter')

parser.add_argument('--lrD', type=float, default=0.00001, help='learning rate, default=0.0002')#调学习率，保证初始是 0.0001
parser.add_argument('--lrG', type=float, default=0.0001, help='learning rate, default=0.0002')

parser.add_argument('--G_Decay_epoch', type=int, default=800, help='learning rate decay epoch, default=100')
parser.add_argument('--D_Decay_epoch', type=int, default=800, help='learning rate decay epoch, default=100')
parser.add_argument('--spe_inter_size', type=int, default=16, help='spetral select inter')

parser.add_argument('--L1_lambda', type=float, default=100, help='lambda for L1 loss')
parser.add_argument('--Cos_lambda', type=float, default=1000, help='lambda for cosine loss')

parser.add_argument('--spe_lambda', type=float, default=1, help='lambda for spectral discriminator loss')
parser.add_argument('--save_epoch', type=int, default=50, help='save model each save_epoch')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')#0.5
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--save_root', required=False, default='results', help='results save path')
opt = parser.parse_args()#解析参数
print(opt)#显示参数

# 创建结果保存路径
root = opt.dataset + '_' + opt.save_root + '/'
model_root=root + 'model/'
model = opt.dataset + '_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(model_root):
    os.mkdir(model_root)

train_dataset = ImageDataset('./data/'+opt.dataset+'/train',opt.crop_size)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)

data = h5py.File(opt.lib_path)
spe_lib = torch.tensor(data['spe_grss']).float().cuda()
num_class = spe_lib.size()[1]

fixed_multi, fixed_hyper = train_loader.__iter__().__next__()
multi_channel=fixed_multi.size()[1]
hyper_channel=fixed_hyper.size()[1]
input_nc=hyper_channel+multi_channel

# network
G = network.generator(network.ResBlock,opt.ngf, multi_channel,num_class)#生成器是一个编解码Unet，kernel_d=ngf

D_spa = network.con_discriminator(opt.ndf, input_nc)
D_spe =network.spe_discriminator(hyper_channel,opt.spe_inter_size,2*opt.ndf)

G.weight_init(mean=0.0, std=0.02)
D_spa.weight_init(mean=0.0, std=0.02)
D_spe.weight_init(mean=0.0, std=0.02)
G.cuda()
D_spa.cuda()
D_spe.cuda()

G.train()
D_spa.train()
D_spe.train()

# loss
BCE_loss = nn.BCELoss().cuda()
L1_loss = nn.L1Loss().cuda()
Cos_loss= nn.CosineSimilarity(dim=1).cuda()

# Adam optimizer
G_optimizer = optim.Adam([{'params':G.parameters(),'initial_lr':0.0001}],lr=opt.lrG, betas=(opt.beta1, opt.beta2))
D_spa_optimizer = optim.Adam([{'params':D_spa.parameters(),'initial_lr':0.00001}], lr=opt.lrD, betas=(opt.beta1, opt.beta2))
D_spe_optimizer = optim.Adam([{'params':D_spe.parameters(),'initial_lr':0.00001}], lr=opt.lrD, betas=(opt.beta1, opt.beta2))

G_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(G_optimizer, T_max=opt.G_Decay_epoch,eta_min=0.0000001,last_epoch=-1)
D_spa_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(D_spa_optimizer, T_max=opt.D_Decay_epoch,eta_min=0.0000001,last_epoch=-1)
D_spe_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(D_spe_optimizer, T_max=opt.D_Decay_epoch,eta_min=0.0000001,last_epoch=-1)

#判断是否存在初始模型
G_dir = os.path.join(model_root, 'generator_param_' + str(opt.start_epoch) + '.pth')
if opt.start_epoch <= opt.D_start_epoch:
    if os.path.exists(G_dir):
        G.load_state_dict(torch.load(G_dir))
        start_epoch = opt.start_epoch
        print('load successful!')
    else:
        start_epoch=0
        print('load failed, training from initial!')
else:
    D_spa_dir = os.path.join(model_root, 'discriminator_spa_param_' + str(opt.start_epoch) + '.pth')
    D_spe_dir = os.path.join(model_root, 'discriminator_spe_param_' + str(opt.start_epoch) + '.pth')
    if os.path.exists(G_dir) and os.path.exists(D_spa_dir) and os.path.exists(D_spe_dir):
        G.load_state_dict(torch.load(G_dir))
        D_spa.load_state_dict(torch.load(D_spa_dir))
        D_spe.load_state_dict(torch.load(D_spe_dir))
        start_epoch =opt.start_epoch
        print('load successful!')
    else:
        start_epoch = 0
        print('load failed, training from initial!')

print('training start!')

logger = Logger(opt.train_epoch, len(train_loader))
num_iters=0

for epoch in range(start_epoch+1,opt.train_epoch):

    num_iter = 0
    for x_, y_ in train_loader:
        x_, y_ = Variable(x_.cuda()), Variable(y_.cuda())
        pos,brightness = G(x_)
        abundance = F.softmax(pos)
        G_result = brightness * \
                   torch.matmul(spe_lib, torch.reshape(abundance.permute(1, 0, 2, 3),(abundance.shape[1], -1))).\
                       reshape(y_.shape[1],y_.shape[0], y_.shape[2], y_.shape[3]).permute( 1, 0, 2, 3)

        # train generator G
        G.zero_grad()#生成器梯度置0
        G_L1_loss = L1_loss(G_result, y_)
        G_cos_loss = 1 - torch.mean(Cos_loss(G_result, y_))

        if epoch > opt.D_start_epoch:
            # train discriminator D
            D_spa.zero_grad()
            D_spe.zero_grad()

            D_real_result = D_spa(x_, y_).squeeze()
            pre_real, pre_result = D_spe(y_, G_result)
            D_pre_result = D_spa(x_, G_result).squeeze()

            D_spa_real_loss = BCE_loss(D_real_result, Variable(torch.ones(D_pre_result.size()).cuda()))  # 真实数据的loss，希望判定真实数据为真实的概率是1，因此概率与1比
            D_spa_fake_loss = BCE_loss(D_pre_result, Variable(torch.zeros(D_pre_result.size()).cuda()))
            D_spa_loss = (D_spa_real_loss + D_spa_fake_loss) * 0.5

            D_spe_real_loss = BCE_loss(pre_real, Variable(torch.ones(pre_real.size()).cuda()))
            D_spe_fake_loss = BCE_loss(pre_result, Variable(torch.zeros(pre_real.size()).cuda()))
            D_spe_loss = (D_spe_real_loss + D_spe_fake_loss) * 0.5

            D_loss = D_spa_loss + opt.spe_lambda * D_spe_loss
            if (num_iter % opt.G_iter == 0):
                D_loss.backward(retain_graph=True)  # loss反传
                D_spa_optimizer.step()  # 优化
                D_spe_optimizer.step()  # 优化
                pre_real, pre_result = D_spe(y_, G_result)
                D_pre_result = D_spa(x_, G_result).squeeze()
            G_spa_loss = BCE_loss(D_pre_result, Variable(torch.ones(D_pre_result.size()).cuda()))
            G_spe_loss = BCE_loss(pre_result, Variable(torch.ones(pre_result.size()).cuda()))
            G_train_loss = G_spa_loss + opt.spe_lambda * G_spe_loss + opt.L1_lambda * G_L1_loss + opt.Cos_lambda * G_cos_loss

        else:  # 不训D，只训G
            G_train_loss = opt.L1_lambda * G_L1_loss + opt.Cos_lambda * G_cos_loss
            G_spa_loss = torch.zeros([1])
            G_spe_loss = torch.zeros([1])
            D_spa_loss = torch.zeros([1])
            D_spe_loss = torch.zeros([1])
            D_spa_real_loss = torch.zeros([1])
            D_spe_real_loss = torch.zeros([1])
            D_spa_fake_loss = torch.zeros([1])
            D_spe_fake_loss = torch.zeros([1])
            D_loss = torch.zeros([1])

        G_train_loss.backward()
        # nn.utils.clip_grad_norm_(G.parameters(),max_norm=1)
        G_optimizer.step()


        num_iter += 1
        num_iters += 1

        logger.log({'D_spa_real_loss': D_spa_real_loss,'D_spa_fake_loss': D_spa_fake_loss, 'D_spa_loss': D_spa_loss,
                    'D_spe_real_loss': D_spe_real_loss,'D_spe_fake_loss': D_spe_fake_loss,'D_spe_loss': D_spe_loss,
                    'D_loss':D_loss,
                    'G_L1_loss': G_L1_loss,'G_cos_loss':G_cos_loss,'G_spe_loss': G_spe_loss,'G_spa_loss': G_spa_loss,
                    'G_train_loss': G_train_loss
                    },
                   images={'MSI': x_, 'real_HSI': y_, 'fake_HSI': G_result})

        if epoch > opt.lrd_start_epoch:
            G_scheduler.step()
            if (epoch > opt.D_start_epoch):
                D_spe_scheduler.step()
                D_spa_scheduler.step()

    if (epoch % opt.save_epoch ==0): #and (epoch>0):
        if epoch > opt.D_start_epoch:
            torch.save(D_spa.state_dict(), model_root + 'discriminator_spa_param_' + str(epoch) + '.pth')
            torch.save(D_spe.state_dict(), model_root + 'discriminator_spe_param_' + str(epoch) + '.pth')

        torch.save(G.state_dict(), model_root + 'generator_param_'+str(epoch)+'.pth')

