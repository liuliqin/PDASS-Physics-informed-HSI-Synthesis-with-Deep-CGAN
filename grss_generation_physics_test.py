import torch, network, argparse, os
import torch.nn.functional as F
import scipy.io as scio
from util import Cal
from datasets import ImageDataset
import gdal
import h5py

def write_img(filename, im_data):#im_geotrans, im_proj,

    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32


    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape


    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=False, default='grss_dfc_2018',  help='')
    parser.add_argument('--lib_path', required=False, default='spe_grss_345.mat', help='spectral lib path')
    parser.add_argument('--ngf', type=int, default=128)
    parser.add_argument('--input_size', type=int, default=256, help='input size')
    parser.add_argument('--model_epoch', default='1400', help='Which model to test?')
    parser.add_argument('--test_flag', default='test_mini',help='train or test_mini')
    parser.add_argument('--abun_flag', default=True, help='save abundance mat')
    parser.add_argument('--cal_performance', default=True, help='cal performance?')
    opt = parser.parse_args()
    print(opt)

    test_dataset = ImageDataset('./data/'+opt.dataset+'/'+opt.test_flag ,opt.input_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    save_root= opt.dataset + '_results/test_results_'+ opt.model_epoch

    if not os.path.isdir(save_root):
        os.mkdir(save_root)
    save_dir= save_root+'/'+opt.test_flag
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    data = h5py.File(opt.lib_path)
    spe_lib = torch.tensor(data['spe_grss']).float().cuda()
    num_class = spe_lib.size()[1]

    fixed_multi, fixed_hyper = test_loader.__iter__().__next__()
    multi_channel=fixed_multi.size()[1]
    hyper_channel=fixed_hyper.size()[1]
    input_nc=multi_channel+hyper_channel

    G = network.generator(network.ResBlock, opt.ngf, multi_channel, num_class)
    G.cuda()

    G_path = opt.dataset + '_results/model/' + 'generator_param_' + opt.model_epoch + '.pth'
    checkpoint=torch.load(G_path)
    G.load_state_dict(checkpoint)
    # G.eval()
    if opt.cal_performance:
        cal_performance=Cal(65535)
        sum_ssim=0
        sum_psnr=0
        sum_sam=0
        sum_rmse=0
        sum_mrae=0
        f = open(save_dir + '/performance.txt', 'w')
    # network
    n = 0
    print('test start!')
    for x_, y_ in test_loader:
        with torch.no_grad():
            x_ = x_.cuda()
        s = test_loader.dataset.multi_path[n][0:-4]
        pos,brightness = G(x_)
        abundance=F.softmax(pos,dim=1)
        if opt.abun_flag:
            abun=abundance[0].cpu().detach().numpy()
            bright=brightness[0].cpu().detach().squeeze().numpy()
            scio.savemat(save_dir+'/'+s+'_abundance.mat', {'abundance': abun,'bright':bright,})
        pre_img=brightness * torch.matmul(spe_lib,torch.reshape(abundance[0],(abundance.shape[1],-1))).reshape(y_.shape)
         # save hyper
        hyper_image = (pre_img[0].cpu().detach().numpy() * 4095).astype('uint16')  # chw
        path = save_dir + '/' + s + '.tif'
        write_img(path, hyper_image)
        #calculating metrics
        hyper_image.astype('float64')
        hyper = (y_[0].cpu().detach().numpy() * 4095).astype('float64')
        if opt.cal_performance:
            mssim=cal_performance.cal_ssim(hyper_image,hyper)
            mpsnr=cal_performance.cal_psnr(hyper_image,hyper)
            rmse,mrae,sam = cal_performance.cal_rmse_mrae_sam(hyper_image,hyper)
            f.write('\n')
            f.write(s)
            f.write('    rmse='+str(rmse)+'   mrae='+str(mrae)+'   sam='+str(sam)+'   ssim='+str(mssim)+'   psnr='+str(mpsnr))
            f.write('\n')
            sum_psnr+=mpsnr
            sum_ssim+=mssim
            sum_sam+=sam
            sum_rmse+=rmse
            sum_mrae+=mrae
        n += 1
        print('%d images generation complete!' % n)
    if opt.cal_performance:
        mean_ssim=sum_ssim/n
        mean_rmse=sum_rmse/n
        mean_mrae=sum_mrae/n
        mean_sam=sum_sam/n
        mean_psnr=sum_psnr/n
        f.write('mean of all images:\n')
        f.write('rmse='+str(mean_rmse)+'   mrae='+str(mean_mrae)+'   sam='+str(mean_sam)+'   ssim='+str(mean_ssim)+'   psnr='+str(mean_psnr))
        f.write('\n')
        f.close()

