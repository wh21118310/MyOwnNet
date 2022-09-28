# -*- coding: utf-8 -*-

"""
@Time : 2022/5/18
@Author : FawkesLee
@File : train_val
@Description :
"""
import glob
import os.path
from os.path import exists

import matplotlib.pyplot as plt
from PIL import Image
from timm.utils import get_state_dict
from torch.autograd import Variable
from torch.cuda import empty_cache, synchronize
from torch.cuda.amp import autocast
from torch.nn import DataParallel
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
# from nets.PFNet import PFNet
# from nets.base import PFNet
# from nets.base_MixTEM_PM import PFNet
from nets.base_MixTEM_PM_UP import PFNet
# from nets.PFNet_ASPP_Mixwise import PFNet
from utils.arguments import *
from utils.data_process import weights_init, ImageFolder
from utils.get_metric import FWIoU, mPA, Precision, Recall, F1
from utils.loss import IoU, structure_loss
from utils.transform import joint_transform, img_transform, target_transform, to_pil

'''Config Information'''
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
seed_torch(seed=2022)
device_ids = [0]  # 设定可用的GPU

'''Loading criterion'''
# ce_loss = CrossEntropyLoss(ignore_index=0)
# '''Diceis used when the distribution of positive and negative samples is extremely unbalanced
#     usually combine ce or Focal'''
# dice_loss = DiceLoss(mode='multiclass', ignore_index=0)
# '''IoU Loss'''
# jaccard_loss = JaccardLoss(mode='multiclass')
# '''use  alpha to adjust teh imblance of positive and negative samples'''
focal_loss = FocalLoss(mode='multiclass', ignore_index=0, alpha=0.5)
# '''Smoothing Jaccard loss'''
# lovasz_loss = LovaszLoss(mode='multiclass', ignore_index=0)
# # criterion = ce_loss.cuda(device=device_ids[0])
# criterion = JointLoss(ce_loss, jaccard_loss).cuda(device=device_ids[0])
# criterion2 = JointLoss(ce_loss, focal_loss).cuda(device=device_ids[0])
bce_loss = BCEWithLogitsLoss().cuda(device=device_ids[0])
iou_loss = IoU().cuda(device=device_ids[0])
struct_Loss = structure_loss().cuda(device=device_ids[0])


def bce_iou_loss(pred, target):
    bce_out = bce_loss(pred, target)
    iou_out = iou_loss(pred, target)
    return bce_out + iou_out


def main(args):
    """Path Settings"""
    save_dir = join(nowPath, 'out', args['model_name'])  # 权值与日志文件保存的文件夹
    save_ckpt_dir, save_log_dir = join(save_dir, 'ckpt'), join(save_dir, 'log')
    save_test_result_dir = join(save_dir, 'results')
    check_path(save_dir), check_path(save_ckpt_dir), check_path(save_log_dir), check_path(save_test_result_dir)
    args['ckpt_dir'], args['log_dir'], args['test_result'] = save_ckpt_dir, save_log_dir, save_test_result_dir
    best_ckpt = save_ckpt_dir + "/best_epoch*.pth"
    get_best = glob.glob(best_ckpt)
    if len(get_best) == 0:
        args['best_ckpt'] = ''
    else:
        args['best_ckpt'] = get_best[0]
    '''Logger'''
    log_name = join(save_log_dir, args["model_name"] + '.log')
    logger = initial_logger(log_name)  # log file

    '''Model Settings'''
    model = PFNet(bk=args['backbone'], model_path=args['backbone_path'])
    if not args['model_init']:
        weights_init(model)
    model = model.cuda(device=device_ids[0])
    '''Loading Datasets'''
    train_imgs_dir, val_imgs_dir = join(args["data_dir"], "train/images"), join(args["data_dir"], "val/images")
    train_labels_dir, val_labels_dir = join(args["data_dir"], "train/gt"), join(args["data_dir"], "val/gt")
    train_data = ImageFolder(imgs_dir=train_imgs_dir, labels_dir=train_labels_dir, joint_transform=joint_transform,
                             image_transform=img_transform, target_transform=target_transform)
    val_data = ImageFolder(imgs_dir=val_imgs_dir, labels_dir=val_labels_dir, joint_transform=joint_transform,
                           image_transform=img_transform, target_transform=target_transform)
    train_data_len, val_data_len = train_data.__len__(), val_data.__len__()
    logger.info('Total Epoch:{}, Training num:{}, Validation num:{}'.format(args['epoch_num'], train_data_len,
                                                                            val_data_len))
    # ------------------------------------------------------------------#
    # num_workers用于设置是否使用多线程读取数据，1代表关闭多线程。开启后会加快数据读取速度，但是会占用更多内存。
    # Windows只可设定为0
    # ------------------------------------------------------------------#
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args["training_batch_size"], num_workers=0,
                              pin_memory=True)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=args["training_batch_size"], num_workers=0,
                            pin_memory=True)
    args['train_loader'] = train_loader
    args['val_loader'] = val_loader
    args['total_epoch'] = args['epoch_num'] * len(train_loader)
    '''EMA Strategy'''
    ema = get_EMA(useEMA=args['model_ema'], net=model)
    '''Loading Optimizer and Scheduler'''
    optimizer = get_optimizer(model=model, args=args)
    '''Loading Scaler'''
    scaler = get_scaler(use_amp=args["use_fp16"])
    '''For Epoch & Restart'''
    # support for the restart from breakpoint
    if args['Resume'] and exists(args['best_ckpt']):
        checkpoint = torch.load(args['best_ckpt'])
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args['epoch_start'] = checkpoint['epoch']
        # if checkpoint['scheduler'] is not None:
        # scheduler.load_state_dict(checkpoint['scheduler'])
        args['total_epoch'] = (args['epoch_num'] - int(args['epoch_start'])) * len(train_loader)
    train_val(model, optimizer, logger, args, scaler, ema)
    test(model, args)


def train_val(model, optimizer, logger, args, scaler=None, ema=None):
    train_loader_len, val_loader_len = len(args['train_loader']), len(args['val_loader'])
    train_loss_total_epochs, valid_loss_total_epochs = list(), list()
    epoch_lr, epoch_iou, epoch_mpa = list(), list(), list()
    '''Train & val'''
    best_iou, best_mpa = .01, .01
    best_epoch, last_index = 0, 0.
    filename = ""
    curr_iter = torch.load(args['best_ckpt'])['iter'] if args['Resume'] else 0
    for epoch in range(args['epoch_start'] + 1, args['epoch_start'] + 1 + args['epoch_num']):
        model.train()
        train_main_loss = AverageMeter()
        train_bar = tqdm(args['train_loader'], total=train_loader_len)
        # scheduler.step()  # when use stepLR\ExponentialLR
        for batch_idx, (image, label) in enumerate(train_bar, start=1):
            if args['poly_train']:
                if args['warmup']:
                    if epoch < args['warmup_epoch']:
                        base_lr = args['warmup_lr'] + (args['lr'] - args['warmup_lr']) / (args['warmup_epoch'] - epoch)
                        args['warmup_lr'] = base_lr
                        optimizer.param_groups[0]['lr'] = 2 * base_lr
                        optimizer.param_groups[1]['lr'] = base_lr
                    if epoch == args['warmup_epoch']:
                        args['warmup'] = False
                        base_lr = args['lr']
                        optimizer.param_groups[0]['lr'] = 2 * base_lr
                        optimizer.param_groups[1]['lr'] = base_lr
                else:
                    base_lr = args['lr'] * (1 - float(curr_iter) / float(args['total_epoch'])) ** args['lr_decay']
                    # // : 整数除法, / :浮点数除法
                    optimizer.param_groups[0]['lr'] = 2 * base_lr
                    optimizer.param_groups[1]['lr'] = base_lr
            optimizer.zero_grad(set_to_none=True)
            image = image.to(device=device_ids[0], dtype=torch.float32, non_blocking=True)
            label = label.to(device=device_ids[0], dtype=torch.float32, non_blocking=True)
            if scaler is None:
                predict_1, predict_2, predict_3, outputs = model(image)
                loss_1 = bce_iou_loss(predict_1, label)
                loss_2 = struct_Loss(predict_2, label)
                loss_3 = struct_Loss(predict_3, label)
                loss_4 = struct_Loss(outputs, label)
                loss = 1 * loss_1 + 1 * loss_2 + 2 * loss_3 + 4 * loss_4
                loss.backward()
                optimizer.step()
            else:
                with autocast():
                    predict_1, predict_2, predict_3, outputs = model(image)
                    loss_1 = bce_iou_loss(predict_1, label)
                    loss_2 = struct_Loss(predict_2, label)
                    loss_3 = struct_Loss(predict_3, label)
                    loss_4 = struct_Loss(outputs, label)
                    loss = 1 * loss_1 + 1 * loss_2 + 2 * loss_3 + 4 * loss_4
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            if args['clip_grad']:  # use clip_grad
                clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
            if ema is not None:  # update EMA
                ema.update(model)
            optimizer.zero_grad(set_to_none=True)

            train_main_loss.update(loss.cpu().detach().numpy())
            train_bar.set_description(desc='[train] epoch:{} iter:{}/{} lr:{:.4f} loss:{:.4f}'.format(
                epoch, batch_idx, train_loader_len, optimizer.param_groups[-1]['lr'],
                train_main_loss.average()))
            if batch_idx == train_loader_len:
                logger.info('[train] epoch:{} iter:{}/{} lr:{:.4f} loss:{:.4f}'.format(
                    epoch, batch_idx, train_loader_len, optimizer.param_groups[-1]['lr'],
                    train_main_loss.average()))
        # scheduler.step(epoch)  # called after every batch update
        # scheduler.step(epoch) # use Poly strategy
        empty_cache()
        synchronize()

        '''Validation '''
        model.eval()
        val_bar = tqdm(args['val_loader'])
        val_loss = AverageMeter()
        acc_meter = AverageMeter()
        fwIoU_meter = AverageMeter()
        mpa_meter = AverageMeter()
        with torch.no_grad():
            for batch_idx, (image, label) in enumerate(val_bar, start=1):
                image = image.to(device=device_ids[0], dtype=torch.float32, non_blocking=True)
                label = label.to(device=device_ids[0], dtype=torch.float32, non_blocking=True)
                predict_1, predict_2, predict_3, predict_4 = model(image)
                loss_1 = bce_iou_loss(predict_1, label)
                loss_2 = struct_Loss(predict_2, label)
                loss_3 = struct_Loss(predict_3, label)
                loss_4 = struct_Loss(predict_4, label)
                loss = 1 * loss_1 + 1 * loss_2 + 2 * loss_3 + 4 * loss_4
                val_loss.update(loss.cpu().detach().numpy())
                val_bar.set_description(desc='[val] epoch:{} iter:{}/{} loss:{:.4f}'.format(
                    epoch, batch_idx, val_loader_len, val_loss.average()))
                if batch_idx == val_loader_len:
                    logger.info('[val] epoch:{} iter:{}/{} loss:{:.4f}'.format(
                        epoch, batch_idx, val_loader_len, val_loss.average()))
                '''以下部分由于数据集的图片是二分类图像，故采用以下方式处理'''
                outputs = predict_4
                outputs = torch.where(outputs >= 0.5, torch.ones_like(outputs), torch.zeros_like(outputs))
                # outputs = torch.argmax(outputs, dim=1)
                outputs = outputs.cpu().detach().numpy()
                for (output, target) in zip(outputs, label):
                    acc = Precision(output, target.cpu(), num_classes=2, ignore_zero=False)
                    mpa = mPA(output, target.cpu(), num_classes=2, ignore_zero=False)
                    fwiou = FWIoU(output, target.cpu(), num_classes=2, ignore_zero=False)
                    acc_meter.update(acc)
                    mpa_meter.update(mpa)
                    fwIoU_meter.update(fwiou)
        # save loss & lr
        train_loss_total_epochs.append(train_main_loss.average())
        valid_loss_total_epochs.append(val_loss.average())
        epoch_lr.append(optimizer.param_groups[-1]['lr'])

        # save Model
        if fwIoU_meter.average() > best_iou:
            model.cpu()
            """model.module.sate_dict() , just used for DataParallel, if no the setting, remove module"""
            state = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'iter': curr_iter}
            if ema is not None:
                state['model_ema'] = get_state_dict(ema)
            if exists(args['best_ckpt']):  # del old
                os.remove(args['best_ckpt'])
            best_ckpt = join(args['ckpt_dir'],
                             'best_epoch{:3d}_fwIoU{:.3f}.pth'.format(epoch, fwIoU_meter.average() * 100))
            torch.save(state, best_ckpt, _use_new_zipfile_serialization=False)
            best_iou = fwIoU_meter.average()
            best_epoch = epoch
            args['best_ckpt'] = best_ckpt
            logger.info('[save] Best Model saved at epoch:{}, fwIou:{}, mPA:{}'.format(best_epoch, fwIoU_meter.average(

            ), mpa_meter.average()))

            if exists(filename):
                os.remove(filename)
            filename = join(args['ckpt_dir'], 'epoch{}_fwiou{:.3f}.pth'.format(epoch, fwIoU_meter.average() * 100))
            torch.save(state, filename, _use_new_zipfile_serialization=False)
            model.cuda(device=device_ids[0])

        if epoch % args['save_inter'] == 0:
            model.cpu()
            state = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            if ema is not None:
                state['model_ema'] = get_state_dict(ema)
            if exists(filename):
                os.remove(filename)
            filename = join(args['ckpt_dir'], 'epoch{}_fwiou{:.3f}.pth'.format(epoch, fwIoU_meter.average() * 100))
            torch.save(state, filename, _use_new_zipfile_serialization=False)
            model.cuda(device=device_ids[0])
        if mpa_meter.average() > best_mpa:
            best_mpa = mpa_meter.average()
        epoch_iou.append(fwIoU_meter.average())
        epoch_mpa.append(mpa_meter.average())
        curr_iter += 1
        # 显示loss
        logger.info("best_epoch:{}, nowIoU: {:.4f}, bestIoU:{:.4f}, now_mPA:{:.4f}, best_mPA:{:.4f}, now_acc:{:.4f}\n"
                    .format(best_epoch, fwIoU_meter.average() * 100, best_iou * 100, mpa_meter.average() * 100,
                            best_mpa * 100, acc_meter.average() * 100))
    indexSet = dict(
        learningRate=epoch_lr,
        FwIoU=epoch_iou,
        mPA=epoch_mpa
    )
    if args['plot']:
        draw(args['epoch_num'], train_loss_total_epochs, valid_loss_total_epochs, indexSet, args['log_dir'])


def test(model, args):
    import numpy as np
    import matplotlib as mpl
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体支持
    assert os.path.exists(args['best_ckpt']), "There is no Weight Files to use"
    params = torch.load(args['best_ckpt'])
    model.load_state_dict(params['model'])
    model.eval()
    with torch.no_grad():
        test_data_dir = os.path.join(args['data_dir'], 'test')
        test_data_images = os.path.join(test_data_dir, 'images')
        test_data_images = glob.glob(test_data_images + "/*.png")
        if args['save_results']:
            fig = plt.figure(figsize=(12, 5))
        for test_data_image in test_data_images:
            label = test_data_image.replace('images', 'gt')
            name = os.path.basename(test_data_image)
            img = Image.open(test_data_image).convert('RGB')
            img_var = Variable(img_transform(img).unsqueeze(0)).cuda(device_ids[0])
            _, _, _, prediction = model(img_var)
            prediction = np.array(to_pil(prediction.data.squeeze(0).cpu()))
            if np.max(prediction) > 1:
                prediction = prediction / 255
            prediction[prediction >= 0.5] = 255
            prediction[prediction < 0.5] = 0
            if args['save_results']:
                original_image = img
                prediction = Image.fromarray(prediction).convert('L')
                label = Image.open(label).convert('L')
                fig.add_subplot(1, 3, 1)
                plt.title('original')
                plt.axis('off')
                plt.imshow(original_image)
                fig.add_subplot(1, 3, 2)
                plt.title('prediction')
                plt.axis('off')
                plt.imshow(prediction)
                fig.add_subplot(1, 3, 3)
                plt.title('label')
                plt.axis('off')
                plt.imshow(label)
                fig.tight_layout()
                fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)  # 调整子图间距
                fig.savefig(os.path.join(args['test_result'], name))
                plt.clf()
        plt.close(fig)


if __name__ == '__main__':
    args = dict(
        # model_name='resNet50_mixwise_PM_UP',
        model_name='resNet50_mixwise_PM_UP_original',
        backbone='resnet50',
        # backbone='swinT_base',
        backbone_path='./params/resnet/resnet50.pth',
        model_init=True,  # if backbone_path is None, Set False Please.

        epoch_num=500,
        training_batch_size=8,  # 以8为基数效果更佳
        data_dir=r"dataset/MarineFarm_80",

        epoch_start=0,
        optimizer='sgd',  # choice: sgd, adam
        lr=1e-3,
        lr_decay=0.9,
        weight_decay=5e-4,
        momentum=0.9,
        poly_train=True,  # if True, use Poly strategy
        warmup=True,  # warmup Strategy
        warmup_epoch=50,
        warmup_lr=1e-4,

        save_inter=10,  # 用来存模型
        Resume=True,  # used for discriminate the status from the breakpoint or start
        model_ema=False,  # if True, use Model Exponential Moving Average
        clip_grad=False,  # if True, gradient clip
        use_fp16=False,  # if True, Mixed Precision Training or AMP(Automatically Mixed Precision)
        plot=True,  # whether draw a picture for show the process of train and validation.
        save_results=True,
    )
    nowPath = os.getcwd()
    main(args=args)
