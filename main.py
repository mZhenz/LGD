#!/usr/bin/env python
import os
import time
import json
import torch.optim
import torch.nn.parallel
import torch.distributed as dist
from tools.opts import parse_opt
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
# from tools.dataset import TSVDataset
from tools.logger import setup_logger
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from tools.utils import simclr_aug, mocov1_aug, mocov2_aug, swav_aug, adjust_learning_rate, \
     soft_cross_entropy, AverageMeter, ValueMeter, ProgressMeter, resume_training, \
     load_simclr_teacher_encoder, load_moco_teacher_encoder, load_swav_teacher_encoder, save_checkpoint, accuracy, \
     save_checkpoint_mod

import clip
# import seed.seed
from clipdistiller.clipdistiller import ClipDistiller
import clipdistiller.models as models
from clip.imagenet_zeroshot_data import imagenet_classnames, openai_imagenet_template
from clip.caltech256_zeroshot_data import caltech256_classnames
from clip.coco_zeroshot_data import coco_classnames, lvis_classname
from clip.citscape_zeroshot_data import cityscapes_classnames
from clip.ade20k_zeroshot_data import ade20k_classnames
from tools.zero_shot_eval import zero_shot_classifier

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def main(args):

    # set-up the output directory
    os.makedirs(args.output, exist_ok=True)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        cudnn.benchmark = True

        # create logger
        logger = setup_logger(output=args.output, distributed_rank=dist.get_rank(),
                              color=False, name="SEED")

        if dist.get_rank() == 0:
            path = os.path.join(args.output, "config.json")
            with open(path, 'w') as f:
                json.dump(vars(args), f, indent=2)
            logger.info("Full config saved to {}".format(path))

        # save the distributed node machine
        logger.info('world size: {}'.format(dist.get_world_size()))
        logger.info('local_rank: {}'.format(args.local_rank))
        logger.info('dist.get_rank(): {}'.format(dist.get_rank()))

    else:
        # create logger
        logger = setup_logger(output=args.output, color=False, name="SEED")
        logger.info('Single GPU mode for debugging.')

    # create model
    logger.info("=> creating student encoder '{}'...".format(args.student_arch))
    logger.info("=> creating teacher encoder '{}'...".format(args.teacher_arch))

    # use SimCLR and SWAV used their customized ResNet architecture with minor differences.
    # if args.teacher_ssl != 'moco':
    #     args.teacher_arch = args.teacher_ssl + '_' + args.teacher_arch

    # some architectures are not supported yet. It needs to be expanded manually.
    assert args.teacher_arch in models.__dict__

    logger.info("=> creating CLIP Model...")
    clip_model, _ = clip.load('RN50', download_root='./clip/weights/', jit=False)

    # create text classifier
    logger.info("=> creating Text Classifier...")
    if args.text == 'imagenet':
        classnames = imagenet_classnames
    elif args.text == 'caltech':
        classnames = caltech256_classnames
    elif args.text == 'coco':
        classnames = coco_classnames
    elif args.text == 'cityscapes':
        classnames = cityscapes_classnames
    elif args.text == 'ade20k':
        classnames = ade20k_classnames
    elif args.text == 'lvis':
        classnames = lvis_classname
    assert args.text in ['imagenet', 'caltech', 'coco', 'cityscapes', 'ade20k', 'lvis']

    classifier = zero_shot_classifier(clip_model, classnames, openai_imagenet_template).float()
    logger.info('=> size of Text Classifier: ' + str(classifier.shape))
    # initialize model object, feed student and teacher into encoders.
    logger.info("=> creating CLIP Distiller...")
    model = ClipDistiller(student=models.__dict__[args.student_arch],
                          teacher=clip_model,
                          classifier=classifier,
                          dim=args.dim,
                          t=args.temp,
                          mlp=args.student_mlp,
                          temp=args.distill_t,
                          m=args.momen,
                          dist=args.distributed,)

    logger.info(model)

    if args.distributed:
        logger.info('=> Entering distributed mode.')

        model = torch.nn.parallel.DistributedDataParallel(model.cuda(),
                                                          device_ids=[args.local_rank],
                                                          broadcast_buffers=False,
                                                          find_unused_parameters=True)

        logger.info('=> Model now distributed.')

        args.lr_mult = args.batch_size / 256
        args.warmup_epochs = 5
        optimizer = torch.optim.SGD(model.parameters(),
                                    args.lr_mult * args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        # tensorboard
        if dist.get_rank() == 0:
            summary_writer = SummaryWriter(log_dir=args.output)
        else:
            summary_writer = None

    else:
        args.lr_mult = 1
        args.warmup_epochs = 5

        model = model.cuda()

        optimizer = torch.optim.SGD(model.parameters(), args.lr,  momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        summary_writer = SummaryWriter(log_dir=args.output)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            model = resume_training(args, model, optimizer, logger)
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # clear unnecessary weights
    torch.cuda.empty_cache()

    # train_dataset = TSVDataset(os.path.join(args.data, 'train.tsv'), augmentation)
    train_dataset = datasets.ImageFolder(os.path.join(args.data, 'train'),
                                         transform=mocov2_aug)
    logger.info('=> Dataset done.')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

        # ensure batch size is dividable by # of GPUs
        assert args.batch_size // dist.get_world_size() == args.batch_size / dist.get_world_size(), \
            'Batch size is not divisible by num of gpus.'

        # create distributed dataloader
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size // dist.get_world_size(),
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(args.data, 'val'),
                                 transforms.Compose([
                                     transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                          std=[0.26862954, 0.26130258, 0.27577711]),
                                 ])),
            batch_size=args.batch_size // dist.get_world_size(),
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True)

    else:
        # create distributed dataloader
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True,
            drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        loss = train(train_loader, model, soft_cross_entropy, optimizer, epoch, args, logger)

        if epoch % args.val_interval == 0 or epoch == args.epochs - 1:
            top1 = validate(val_loader, model, soft_cross_entropy, args, logger, classifier)

        if summary_writer is not None:
            # Tensor-board logger
            summary_writer.add_scalar('train_loss', loss, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if dist.get_rank() == 0:
            file_str = 'Teacher_{}_T-Epoch_{}_Student_{}_distill-Epoch_{}-checkpoint_{:04d}.pth.tar'\
                .format(args.teacher_ssl, args.epochs, args.student_arch, args.teacher_arch, epoch)
            prefile_str = 'Teacher_{}_T-Epoch_{}_Student_{}_distill-Epoch_{}-checkpoint_{:04d}.pth.tar' \
                .format(args.teacher_ssl, args.epochs, args.student_arch, args.teacher_arch, epoch-1)

            save_checkpoint_mod(
                {
                'epoch': epoch + 1,
                'arch': args.student_arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                },
                is_best=True,
                filename=os.path.join(args.output, file_str),
                prefile=os.path.join(args.output, prefile_str)
            )

            logger.info('==============> checkpoint saved to {}'.format(os.path.join(args.output, file_str)))


def train(train_loader, model, criterion, optimizer, epoch, args, logger):
    batch_time = AverageMeter('Batch Time', ':5.3f')
    data_time = AverageMeter('Data Time', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    lr = ValueMeter('LR', ':5.3f')
    mem = ValueMeter('GPU Memory Used', ':5.0f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, lr, losses, mem],
        prefix="Epoch: [{}]".format(epoch))

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    lr.update(get_learning_rate(optimizer))
    mem.update(torch.cuda.max_memory_allocated(device=0) / 1024.0 / 1024.0)

    # switch to train mode
    model.train()

    # make key-encoder at eval to freeze BN
    if args.distributed:
        model.module.teacher.eval()

        # check the sanity of key-encoder
        for name, param in model.module.teacher.named_parameters():
            if param.requires_grad:
                logger.info("====================> Key-encoder Sanity Failed, parameters are not frozen.")

    else:
        model.teacher.eval()

        # check the sanity of key-encoder
        for name, param in model.teacher.named_parameters():
           if param.requires_grad:
                logger.info("====================> Key-encoder Sanity Failed, parameters are not frozen.")

    end = time.time()

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for i, (images, _) in enumerate(train_loader):

        if not args.distributed:
            images = images.cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        with torch.cuda.amp.autocast(enabled=True):

            logit_img, label_img, logit_lan, label_lan, _, _ = model(image=images)
            loss_i = criterion(logit_img, label_img)
            loss_l = criterion(logit_lan, label_lan)
            loss = 0.5 * (loss_i + loss_l)

        losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, logger)

    return losses.avg


def validate(val_loader, model, criterion, args, logger, classifier):
    batch_time = AverageMeter('Time', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    t_top1 = AverageMeter('T_Acc@1', ':5.2f')
    s_top1 = AverageMeter('S_Acc@1', ':5.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, t_top1, s_top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            target = target.cuda()

            # compute output
            logit_img, label_img, logit_lan, label_lan, s_emb, t_emb = model(images, inference=True)
            loss_i = criterion(logit_img, label_img)
            loss_l = criterion(logit_lan, label_lan)
            loss = 0.5 * (loss_i + loss_l)

            with torch.no_grad():
                logit_s = 100.0 * s_emb @ classifier.float()
                logit_t = 100.0 * t_emb @ classifier.float()

            # measure accuracy and record loss
            acc1_s, _ = accuracy(logit_s, target, topk=(1, 5))
            acc1_t, _ = accuracy(logit_t, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            s_top1.update(acc1_s[0], images.size(0))
            t_top1.update(acc1_t[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i, logger)

        # TODO: this should also be done with the ProgressMeter
        logger.info(' * S_Acc@1 {s_top1.avg:.3f} T_Acc@1 {t_top1.avg:.3f}'.format(s_top1=s_top1, t_top1=t_top1))

    return s_top1.avg


if __name__ == '__main__':
    main(parse_opt())
