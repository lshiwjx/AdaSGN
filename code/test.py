# encoding: utf-8

from __future__ import print_function
import argparse, os, pickle, sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from model.predictor import ConditionalMotionNet, ConditionalAppearanceNet
from model.encoder import define_E
from util import generateLoop, videoWrite, normalize, denormalize


class AnimatingLandscape():

    def __init__(self, args):
        print(args)
        self.model_path = args.model_path
        self.model_epoch = args.model_epoch
        self.gpu = int(args.gpu)
        self.input_path = args.input
        self.outdir_path = args.outdir
        self.t_m = float(args.motion_latent_code)
        self.s_m = float(args.motion_speed)
        self.t_a = float(args.appearance_latent_code)
        self.s_a = float(args.appearance_speed)
        self.t_m = min(1., max(0., self.t_m))
        self.s_m = min(1., max(1e-3, self.s_m))
        self.t_a = min(1., max(0., self.t_a))
        self.s_a = min(1., max(1e-3, self.s_a))
        self.TM = int(args.motion_frame_number)
        self.w, self.h = 256, 256  # Image size for network input
        self.fw, self.fh = None, None  # Output image size
        self.pad = 64  # Reflection padding size for sampling outside of the image

    def PredictMotion(self):
        print('Motion: ')
        # 模型加载
        P_m = ConditionalMotionNet()
        param = torch.load(self.model_path + '/PMNet_weight_' + self.model_epoch + '.pth')
        P_m.load_state_dict(param)
        if self.gpu > -1:
            P_m.cuda(self.gpu)
        # codebook加载  【100，8】
        with open(self.model_path + '/codebook_m_' + self.model_epoch + '.pkl', 'rb') as f:
            codebook_m = pickle.load(f) if sys.version_info[0] == 2 else pickle.load(f, encoding='latin1')
        # 计算选择的是第几个code，由于是小数，差值得到code值
        id1 = int(np.floor((len(codebook_m) - 1) * self.t_m))
        id2 = int(np.ceil((len(codebook_m) - 1) * self.t_m))
        z_weight = (len(codebook_m) - 1) * self.t_m - np.floor((len(codebook_m) - 1) * self.t_m)
        z_m = (1. - z_weight) * codebook_m[id1:id1 + 1] + z_weight * codebook_m[id2:id2 + 1]
        z_m = Variable(torch.from_numpy(z_m.astype(np.float32)))
        if self.gpu > -1:
            z_m = z_m.cuda(self.gpu)
        # 横向与纵向的-1到1
        initial_coordinate = np.array([np.meshgrid(np.linspace(-1, 1, self.w + 2 * self.pad),
                                                   np.linspace(-1, 1, self.h + 2 * self.pad), sparse=False)]).astype(
            np.float32)
        initial_coordinate = Variable(torch.from_numpy(initial_coordinate))
        if self.gpu > -1:
            initial_coordinate = initial_coordinate.cuda(self.gpu)

        with torch.no_grad():
            test_img = cv2.imread(self.input_path)
            test_img = cv2.resize(test_img, (self.w, self.h))  # 256
            test_input = np.array([normalize(test_img)])  # -1～1
            test_input = Variable(torch.from_numpy(test_input.transpose(0, 3, 1, 2)))
            if self.gpu > -1:
                test_input = test_input.cuda(self.gpu)
            padded_test_input = F.pad(test_input, (self.pad, self.pad, self.pad, self.pad), mode='reflect')  # 镜像pad
            # 输出图像大小
            test_img_large = cv2.imread(self.input_path)
            if self.fw == None or self.fh == None:
                self.fh, self.fw = test_img_large.shape[:2]
            test_img_large = cv2.resize(test_img_large, (self.fw, self.fh))
            padded_test_input_large = np.array([normalize(test_img_large)])
            padded_test_input_large = Variable(torch.from_numpy(padded_test_input_large.transpose(0, 3, 1, 2)))
            if self.gpu > -1:
                padded_test_input_large = padded_test_input_large.cuda(self.gpu)
            scaled_pads = (int(self.pad * self.fh / float(self.h)), int(self.pad * self.fw / float(self.w)))
            padded_test_input_large = F.pad(padded_test_input_large,
                                            (scaled_pads[1], scaled_pads[1], scaled_pads[0], scaled_pads[0]),
                                            mode='reflect')

            V_m = list()
            V_f = list()
            old_correpondence = None
            for t in range(self.TM):
                sys.stdout.write("\rProcessing frame %d, " % (t + 1))
                sys.stdout.flush()

                flow = P_m(test_input, z_m) # 【256，256】
                flow[:, 0, :, :] = flow[:, 0, :, :] * (self.w / float(self.pad * 2 + self.w))  # 运动范围调整至pad后
                flow[:, 1, :, :] = flow[:, 1, :, :] * (self.h / float(self.pad * 2 + self.h))
                flow = F.pad(flow, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
                flow = self.s_m * flow  # 根据运动速度缩放flow
                correspondence = initial_coordinate + flow  # 计算运动后的坐标 1，2，384，384
                # 一直都是从原图计算
                if old_correpondence is not None:
                    correspondence = F.grid_sample(old_correpondence, correspondence.permute(0, 2, 3, 1),
                                                   padding_mode='border', align_corners=True)
                # 差值到原图大小，每个点代表当前点在原图的位置
                correspondence_large = F.interpolate(correspondence,
                                                  size=(self.fh + scaled_pads[0] * 2, self.fw + scaled_pads[1] * 2),
                                                  mode='bilinear', align_corners=True)
                y_large = F.grid_sample(padded_test_input_large, correspondence_large.permute(0, 2, 3, 1),
                                        padding_mode='border', align_corners=True)  # 根据输入和flow计算输出
                outimg = y_large.data.cpu().numpy()[0].transpose(1, 2, 0)
                outimg = denormalize(outimg)
                outimg = outimg[scaled_pads[0]:outimg.shape[0] - scaled_pads[0],
                         scaled_pads[1]:outimg.shape[1] - scaled_pads[1]]
                V_m.append(outimg)
                # flow图生成
                # outflowimg = flow.data.cpu().numpy()[0].transpose(1, 2, 0)
                # outflowimg = outflowimg[self.pad:outflowimg.shape[0] - self.pad,
                #              self.pad:outflowimg.shape[1] - self.pad]
                # mag, ang = cv2.cartToPolar(outflowimg[..., 1], outflowimg[..., 0])
                # hsv = np.zeros_like(test_img)
                # hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                # hsv[..., 0] = ang * 180 / np.pi / 2
                # hsv[..., 2] = 255
                # outflowimg = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                # outflowimg = cv2.resize(outflowimg, (self.fw, self.fh))
                # V_f.append(outflowimg)
                # 更新网络的输入图像，网络是每次输入移动后的图，但最终的移动是基于最初的图
                y = F.grid_sample(padded_test_input, correspondence.permute(0, 2, 3, 1), padding_mode='border', align_corners=True)
                test_input = y[:, :, self.pad:y.shape[2] - self.pad, self.pad:y.shape[3] - self.pad]
                old_correpondence = correspondence

            V_mloop = V_m  # generateLoop(V_m)

        return V_mloop, V_f

    def GenerateVideo(self):
        V_mloop, V_f = self.PredictMotion()
        # import matplotlib.pyplot as plt

        videoWrite(V_mloop, out_path=self.outdir_path + '/' + os.path.splitext(self.input_path)[0].split('/')[
            -1] + '_motion.avi')
        # videoWrite(V_f,
        #            out_path=self.outdir_path + '/' + os.path.splitext(self.input_path)[0].split('/')[-1] + '_flow.avi')

        # V = self.PredictAppearance(V_mloop)
        videoWrite(V, out_path=self.outdir_path + '/' + os.path.splitext(self.input_path)[0].split('/')[-1] + '.avi')
        print('\nDone.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AnimatingLandscape')
    parser.add_argument('--model_path', default='../experiments/datanew_wd0_lr1e4_beta5_size128_beta1-test/')
    parser.add_argument('--model_epoch', default='3399')
    parser.add_argument('--gpu', default=0)
    # ../../../data//images/selfie_female_1587984129_01163a147-23.jpg
    parser.add_argument('--input', '-i', default='../data/images/selfie_female_1587984129_01163a147-23.jpg')
    # parser.add_argument('--input', '-i', default='../vis/test_data/human/human_1.jpg')
    parser.add_argument('--motion_latent_code', '-mz', default=np.random.rand())
    parser.add_argument('--motion_speed', '-ms', default=0.5)
    parser.add_argument('--appearance_latent_code', '-az', default=np.random.rand())
    parser.add_argument('--appearance_speed', '-as', default=0.1)
    parser.add_argument('--motion_frame_number', '-mn', default=299)
    parser.add_argument('--outdir', '-o', default='../experiments/datanew_wd0_lr1e4_beta5_size128_beta1-test/')
    args = parser.parse_args()

    AS = AnimatingLandscape(args)
    AS.GenerateVideo()
