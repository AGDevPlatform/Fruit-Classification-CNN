import os
import argparse
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models.Nhom12Mang1 import *
from models.cross_entropy import LabelSmoothingCrossEntropy
import writeLogAcc as wA

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# Remove GPU-related arguments or set them to None
parser.add_argument('-r', '--data', type=str,
                    default='./dataset', help='path to dataset')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int, nargs='+',
                    help='seed for initializing training. ')
parser.add_argument('--action', default='', type=str,
                    help='other information.')


def main():
    global args
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    args.distributed = False
    torch.autograd.set_detect_anomaly(True)

    args.arch = 'Apple_banana_orange'
    filenameLOG = "./checkpoints/%s/" % (args.arch +
                                         '_' + args.action) + '/' + args.arch + '.txt'

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        model = Net(10)

    # Move model to CPU
    model = torch.nn.DataParallel(model)

    print(model)

    print('Number of models parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # Move loss functions to CPU
    train_loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        pathcheckpoint = "./checkpoints/%s/" % (
            args.arch + '_' + args.action) + "model_best.pth.tar"
        if os.path.isfile(pathcheckpoint):
            print("=> loading checkpoint '{}'".format(pathcheckpoint))
            checkpoint = torch.load(pathcheckpoint, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(pathcheckpoint))
            return

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'test')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        m = time.time()
        _, _ = validate(val_loader, model, criterion)
        n = time.time()
        print((n-m)/3600)
        return

    directory = "checkpoints/%s/" % (args.arch + '_' + args.action)
    if not os.path.exists(directory):
        os.makedirs(directory)

    Loss_plot = {}
    train_prec1_plot = {}
    train_prec5_plot = {}
    val_prec1_plot = {}
    val_prec5_plot = {}
    epoch_max = None
    best_prec1 = 0

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        adjust_learning_rate(optimizer, epoch)

        loss_temp, train_prec1_temp, train_prec5_temp = train(
            train_loader, model, train_loss_fn, optimizer, epoch)

        Loss_plot[epoch] = loss_temp
        train_prec1_plot[epoch] = train_prec1_temp
        train_prec5_plot[epoch] = train_prec5_temp

        prec1, prec5 = validate(val_loader, model, criterion)
        val_prec1_plot[epoch] = prec1
        val_prec5_plot[epoch] = prec5

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)

        data_save(directory + 'Loss_plot.txt', Loss_plot)
        data_save(directory + 'train_prec1.txt', train_prec1_plot)
        data_save(directory + 'train_prec5.txt', train_prec5_plot)
        data_save(directory + 'val_prec1.txt', val_prec1_plot)
        data_save(directory + 'val_prec5.txt', val_prec5_plot)

        line = 'Epoch {}/{} summary: loss_train={:.5f}, acc_train={:.2f}%, loss_val={:.2f}, acc_val={:.2f}% (best: {:.2f}% @ epoch {})'.format(
            epoch, args.epochs, loss_temp, train_prec1_temp, 0, prec1, best_prec1, epoch_max)
        wA.writeLogAcc(filenameLOG, line)
        end_time = time.time()
        time_value = (end_time - start_time) / 3600
        print("-" * 80)
        print(time_value)
        print("-" * 80)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses_batch = {}
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        output = model(input)
        loss = criterion(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 2))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            output = model(input)
            loss = criterion(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "checkpoints/%s/" % (args.arch + '_' + args.action)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory + 'model_best.pth.tar')


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous(
            ).view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def data_save(root, file):
    # Tạo file mới nếu chưa tồn tại
    if not os.path.exists(root):
        with open(root, 'w') as f:
            pass

    # Đọc dữ liệu hiện có
    try:
        with open(root, 'r') as file_temp:
            lines = file_temp.readlines()

        # Xác định epoch cuối cùng
        if not lines:
            epoch = -1
        else:
            epoch = int(lines[-1].split()[0])

        # Ghi thêm dữ liệu mới
        with open(root, 'a') as file_temp:
            for line in file:
                if line > epoch:
                    file_temp.write(f"{line} {file[line]}\n")

    except Exception as e:
        print(f"Error in data_save: {e}")
        # Tạo file mới nếu có lỗi đọc file
        with open(root, 'w') as file_temp:
            for line in file:
                file_temp.write(f"{line} {file[line]}\n")


if __name__ == '__main__':
    main()
