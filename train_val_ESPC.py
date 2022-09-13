# -*- coding: utf-8 -*-

"""
@Time : 2022/5/18
@Author : FawkesLee
@File : train_val
@Description :
"""
from os.path import exists

from timm.utils import get_state_dict
from torch.backends import cudnn
from torch.cuda.amp import autocast
from torch.nn import DataParallel
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from nets.PFNet_ASPPMixwise_TailESPCN import PFNet
from utils.arguments import *
from utils.data_process import weights_init, ImageFolder
from utils.get_metric import binary_accuracy, Acc, FWIoU
from utils.loss import IoU, structure_loss
from utils.transform import joint_transform, img_transform, target_transform

'''Config Information'''
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
cudnn.benchmark = True
seed_torch(seed=2022)
device_ids = [0]  # 设定可用的GPU

'''Loading criterion'''
# criterion_name = "bcew"
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
    best_ckpt = join(save_ckpt_dir, 'best_model.pth')  # save the best weights
    check_path(save_dir), check_path(save_ckpt_dir), check_path(save_log_dir)
    args['ckpt_dir'], args['log_dir'] = save_ckpt_dir, save_log_dir
    args['best_ckpt'] = best_ckpt
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
    val_loader = DataLoader(val_data, shuffle=True, batch_size=int(args["training_batch_size"] * 1.2), num_workers=0,
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
    if args['Resume'] and exists(best_ckpt):
        checkpoint = torch.load(best_ckpt)
        model.load_state_dict(checkpoint['backbone'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start = checkpoint['epoch']
        # if checkpoint['scheduler'] is not None:
        # scheduler.load_state_dict(checkpoint['scheduler'])
        args['total_epoch'] = (args['epoch_num'] - int(epoch_start)) * len(train_loader)
    model = DataParallel(model, device_ids=device_ids)
    print("Using {} GPU(s) to Train.".format(len(device_ids)))
    train_val(model, optimizer, logger, args, scaler, ema)


def train_val(model, optimizer, logger, args, scaler=None, ema=None):
    train_loader_len, val_loader_len = len(args['train_loader']), len(args['val_loader'])
    train_loss_total_epochs, valid_loss_total_epochs = list(), list()
    epoch_lr, epoch_iou, epoch_mpa = list(), list(), list()
    '''Train & val'''
    best_iou, best_mpa = .1, .1
    best_epoch, last_index = 0, 0.
    filename = ""
    curr_iter = 0
    for epoch in range(args['epoch_start'] + 1, args['epoch_start'] + 1 + args['epoch_num']):
        model.train()
        train_main_loss = AverageMeter()
        train_bar = tqdm(args['train_loader'], total=train_loader_len)
        # scheduler.step()  # when use stepLR\ExponentialLR
        for batch_idx, (image, label) in enumerate(train_bar, start=1):
            if args['poly_train']:
                base_lr = args['lr'] * (1 - float(curr_iter) / float(args['total_epoch'])) ** args['lr_decay']
                # // : 整数除法, / :浮点数除法
                optimizer.param_groups[0]['lr'] = 2 * base_lr
                optimizer.param_groups[1]['lr'] = base_lr
            optimizer.zero_grad(set_to_none=True)
            batch_size = image.size(0)
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

            train_main_loss.update(loss.cpu().detach().numpy(), num=batch_size)
            train_bar.set_description(desc='[train] epoch:{} iter:{}/{} lr:{:.4f} loss:{:.4f}'.format(
                epoch, batch_idx, train_loader_len, optimizer.param_groups[-1]['lr'],
                train_main_loss.average()))
            if batch_idx == train_loader_len:
                logger.info('[train] epoch:{} iter:{}/{} lr:{:.4f} loss:{:.4f}'.format(
                    epoch, batch_idx, train_loader_len, optimizer.param_groups[-1]['lr'],
                    train_main_loss.average()))
        # scheduler.step(epoch)  # called after every batch update
        # scheduler.step(epoch) # use Poly strategy
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        '''Validation '''
        model.eval()
        val_bar = tqdm(args['val_loader'])
        val_loss = AverageMeter()
        acc_meter = AverageMeter()
        fwIoU_meter = AverageMeter()
        mpa_meter = AverageMeter()
        with torch.no_grad():
            for batch_idx, (image, label) in enumerate(val_bar, start=1):
                batch_size = image.size(0)
                image = image.to(device=device_ids[0], dtype=torch.float32, non_blocking=True)
                label = label.to(device=device_ids[0], dtype=torch.float32, non_blocking=True)
                predict_1, predict_2, predict_3, predict_4 = model(image)
                loss_1 = bce_iou_loss(predict_1, label)
                loss_2 = struct_Loss(predict_2, label)
                loss_3 = struct_Loss(predict_3, label)
                loss_4 = struct_Loss(predict_4, label)
                loss = 1 * loss_1 + 1 * loss_2 + 2 * loss_3 + 4 * loss_4
                val_loss.update(loss.cpu().detach().numpy(), num=batch_size)
                val_bar.set_description(desc='[val] epoch:{} iter:{}/{} loss:{:.4f}'.format(
                    epoch, batch_idx, val_loader_len, val_loss.average()))
                if batch_idx == val_loader_len:
                    logger.info('[val] epoch:{} iter:{}/{} loss:{:.4f}'.format(
                        epoch, batch_idx, val_loader_len, val_loss.average()))
                '''以下部分由于数据集的图片是二分类图像，故采用以下方式处理'''
                # outputs = torch.where(outputs > 0.5, torch.ones_like(outputs), torch.zeros_like(outputs))
                outputs = predict_4.cpu().detach().numpy()
                for (outputs, target) in zip(outputs, label):
                    acc, valid_sum = binary_accuracy(outputs, target)
                    mpa = Acc(outputs.squeeze(), target.cpu().squeeze(), ignore_zero=False)
                    fwiou = FWIoU(outputs.squeeze(), target.cpu().squeeze(), ignore_zero=False)
                    acc_meter.update(acc, num=batch_size)
                    mpa_meter.update(mpa, num=batch_size)
                    fwIoU_meter.update(fwiou, num=batch_size)
        # save loss & lr
        train_loss_total_epochs.append(train_main_loss.average())
        valid_loss_total_epochs.append(val_loss.average())
        epoch_lr.append(optimizer.param_groups[-1]['lr'])
        curr_iter += 1
        # save Model
        if fwIoU_meter.average() > best_iou:
            model.cpu()
            """model.module.sate_dict() , just used for DataParallel, if no the setting, remove module"""
            state = {'epoch': epoch, 'model': model.module.state_dict(), 'optimizer': optimizer.state_dict()}
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
            state = {'epoch': epoch, 'model': model.module.state_dict(), 'optimizer': optimizer.state_dict()}
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
        # 显示loss
        logger.info("best_epoch:{}, nowIoU: {:.4f}, bestIoU:{:.4f}, now_mPA:{:.4f}, best_mPA:{:.4f}\n"
                    .format(best_epoch, fwIoU_meter.average() * 100, best_iou * 100, mpa_meter.average() * 100,
                            best_mpa * 100))
    indexSet = dict(
        learningRate=epoch_lr,
        FwIoU=epoch_iou,
        mPA=epoch_mpa
    )
    if args['plot']:
        draw(args['epoch_num'], train_loss_total_epochs, valid_loss_total_epochs, indexSet, args['log_dir'])


if __name__ == '__main__':
    args = dict(
        model_name='PFNet_resnet50_MixWise_ESPCN1',
        backbone='resnet50',
        backbone_path="./params/resnet/resnet50.pth",
        model_init=True,  # if backbone_path is None, Set False Please.

        epoch_num=500,
        training_batch_size=2,  # 以8为基数效果更佳
        data_dir=r"dataset/MarineFarm_80",

        epoch_start=0,
        optimizer='sgd',  # choice: sgd, adam
        lr=1e-3,
        lr_decay=0.9,
        weight_decay=5e-4,
        momentum=0.9,
        # snapshot='',
        poly_train=True,  # if True, use Poly strategy

        save_inter=10,  # 用来存模型
        Resume=False,  # used for discriminate the status from the breakpoint or start
        model_ema=False,  # if True, use Model Exponential Moving Average
        clip_grad=False,  # if True, gradient clip
        use_fp16=False,  # if True, Mixed Precision Training or AMP(Automatically Mixed Precision)
        plot=True  # whether draw a picture for show the process of train and validation.
    )
    nowPath = os.getcwd()
    main(args=args)
