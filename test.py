# -*- coding: utf-8 -*-

"""
    @Time : 2022/8/21 10:23
    @Author : FaweksLee
    @Email : 121106010719@njust.edu.cn
    @File : test
    @Description : Test code for project
"""
import glob
import os
from os.path import join

import numpy as np
import torch.cuda
from PIL import Image
from torch.autograd import Variable
from torch.backends import cudnn
from torchvision.transforms import transforms

from nets.PFNet_ASPP_Mixwise import PFNet
from utils.arguments import seed_torch, check_path

seed_torch(seed=2022)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
cudnn.benchmark = False
device_ids = [1]
torch.cuda.set_device(device_ids[0])

img_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])
to_pil = transforms.ToPILImage()


def main(args):
    print(torch.__version__)

    # model = PFNet(bk=args['backbone_name'], model_path=args['backbone_path'])
    model = PFNet(bk=args['backbone_name'])
    model = model.cuda(device=device_ids[0])
    '''Save Path'''
    Params = torch.load(args['best_ckpt'])
    # '''Main Iteration'''
    # epoch_iou = list()  # save index
    '''Test'''
    model.load_state_dict(Params['model'])
    model.eval()
    with torch.no_grad():
        '''Loading Datasets'''
        test_data = args['data_dir'] + "/*.png"
        test_datas = glob.glob(test_data)
        if args['save_results']:
            check_path(args['result_path'])
        for test_data in test_datas:
            idx = os.path.basename(test_data).split(".")[0]
            img = Image.open(test_data).convert("RGB")
            w, h = img.size
            img_var = Variable(img_transform(img).unsqueeze(0)).cuda(device_ids[0])
            _, _, _, prediction = model(img_var)
            # prediction = np.array(transforms.Resize((h, w))(to_pil(prediction.data.squeeze(0).cpu())))
            prediction = np.array(to_pil(prediction.data.squeeze(0).cpu()))
            if args['save_results']:
                Image.fromarray(prediction).convert("L").save(os.path.join(args['result_path'], idx+".png"))
                print("save Image :", idx)


if __name__ == '__main__':
    args = dict(
        model_name='PFNet_resnet50_MixWise',
        backbone_path='./params/resnet/resnet50.pth',
        backbone_name='resnet50',
        data_dir=r"dataset/MarineFarm_80/test/images",
        save_results=True,
    )
    nowPath = os.getcwd()
    result_dir = join(nowPath, 'out', args['model_name'], 'results')  # 结果存储
    check_path(result_dir)
    pth_load_path = glob.glob(os.path.join(nowPath, 'out', args['model_name'], 'ckpt') + "/best_epoch*.pth")
    args['result_path'] = result_dir
    assert len(pth_load_path) >= 1, "There is no Weight Files to use"
    args['best_ckpt'] = pth_load_path[0]
    main(args)
