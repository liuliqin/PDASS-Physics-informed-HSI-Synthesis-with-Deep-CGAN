import itertools, imageio, torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from torchvision import datasets
from scipy.misc import imresize
from visdom import Visdom
from sklearn.metrics.pairwise import cosine_similarity
import sys
import time
import datetime

class Cal():
    def __init__(self,num_range):
        self.num_range=num_range
    def cal_ssim(self,img1,img2):
        '''calculate SSIM
        the same outputs as MATLAB's
        img1, img2: [0, 255]
        '''
        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')
        if img1.ndim == 2:
            return self.ssim(img1, img2)
        elif img1.ndim == 3:
            ssims = []
            for i in range(img1.shape[0]):
                ssims.append(self.ssim(img1[i, :, :], img2[i, :, :]))  # 改
            return np.array(ssims).mean()
        else:
            raise ValueError('Wrong input image dimensions.')
    def ssim(self,img1,img2):

        C1 = (0.01 * self.num_range) ** 2
        C2 = (0.03 * self.num_range) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()
    def cal_psnr(self,img1,img2):
        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')
        if img1.ndim == 2:
            return self.psnr(img1, img2)
        elif img1.ndim == 3:
            sum_psnr = 0
            for i in range(img1.shape[0]):
                this_psnr = self.psnr(img1[i, :, :], img2[i, :, :])
                sum_psnr += this_psnr
        return sum_psnr / img1.shape[0]
    def psnr(self,img1,img2):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(self.num_range *1.0 / math.sqrt(mse))
    def cal_rmse_mrae_sam(self,img1,img2):# img1 result img2 label
        channel, height, width = img2.shape
        sum_se = 0
        sum_mrae = 0
        sum_sam = 0
        for i in range(0, height):
            for j in range(0, width):
                sum_se += np.sum((img1[:, i, j] - img2[:, i, j]) ** 2)
                A = img2[:, i, j]
                sum_mrae += np.sum(abs(img1[:, i, j] - img2[:, i, j]) / (img2[:, i, j] + 1))
                spe_res = img1[:, i, j].reshape(1, -1)
                spe_lab = img2[:, i, j].reshape(1, -1)
                sum_sam += math.acos(cosine_similarity(spe_lab, spe_res))

        rmse = (sum_se / (height * width * channel)) ** 0.5
        mrae = sum_mrae / (height * width * channel)
        sam = sum_sam / (height * width)
        return rmse,mrae,sam
    def SID(self,x, y):
        p = np.zeros_like(x, dtype=np.float)
        q = np.zeros_like(y, dtype=np.float)
        Sid = 0
        for i in range(len(x)):
            p[i] = x[i] / np.sum(x)
            q[i] = y[i] / np.sum(y)
        for j in range(len(x)):
            Sid += p[j] * np.log10(p[j] / q[j]) + q[j] * np.log10(q[j] / p[j])
        return Sid
    def SAM(self,x, y):# 计算SAM
        s = np.sum(np.dot(x, y))
        t = np.sqrt(np.sum(x ** 2)) * np.sqrt(np.sum(y ** 2))
        th = np.arccos(s / t)
        # print(s,t)
        return th
def tensor2image(tensor):
    imtensor=tensor[0]
    if imtensor.size()[0]== 3:
        imbinar = imtensor
    elif imtensor.size()[0]<3:
        imbinar=np.tile(imtensor[0],(3,1,1))
    else:
        imbinar = imtensor[[5, 12, 23], :, :]
    image = 127.5 * (imbinar.cpu().float().numpy() + 1.0)
    return image.astype(np.uint8)


class Logger():
    def __init__(self, n_epochs, batches_epoch):
        self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}


    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            # print(tensor.requires_grad)
            if image_name not in self.image_windows:

                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.detach()), opts={'title':image_name})
            else:
                self.viz.image(tensor2image(tensor.detach()), win=self.image_windows[image_name], opts={'title':image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]),
                                                                    opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1



#def show_result(G, x_, y_, num_epoch, show = False, save = False, path = 'result.png'):
    # G.eval()
    # test_images = G(x_)
    #
    # size_figure_grid = 3
    # fig, ax = plt.subplots(x_.size()[0], size_figure_grid, figsize=(5, 5))
    # for i, j in itertools.product(range(x_.size()[0]), range(size_figure_grid)):
    #     ax[i, j].get_xaxis().set_visible(False)
    #     ax[i, j].get_yaxis().set_visible(False)
    #
    # for i in range(x_.size()[0]):
    #     ax[i, 0].cla()
    #     ax[i, 0].imshow((x_[i].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
    #     ax[i, 1].cla()
    #     ax[i, 1].imshow((test_images[i].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
    #     ax[i, 2].cla()
    #     ax[i, 2].imshow((y_[i].numpy().transpose(1, 2, 0) + 1) / 2)
    #
    # label = 'Epoch {0}'.format(num_epoch)
    # fig.text(0.5, 0.04, label, ha='center')
    #
    # if save:
    #     plt.savefig(path)
    #
    # if show:
    #     plt.show()
    # else:
    #     plt.close()

