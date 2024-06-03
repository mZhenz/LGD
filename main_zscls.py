import os
import torch
from torchvision import datasets, transforms
from tqdm import tqdm

import clip
from clip.imagenet_zeroshot_data import imagenet_classnames, openai_imagenet_template
from clip.caltech256_zeroshot_data import caltech256_classnames
from tools.zero_shot_eval import zero_shot_classifier
from tools.opts import parse_opt
from tools.utils import AverageMeter, soft_cross_entropy, accuracy
from clipdistiller.clipdistiller import ClipDistiller
import clipdistiller.models as models


def validate(val_loader, model, criterion, classifier):
    losses = AverageMeter('Loss', ':5.3f')
    t_top1 = AverageMeter('T_Acc@1', ':5.2f')
    s_top1 = AverageMeter('S_Acc@1', ':5.2f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for (images, target) in tqdm(val_loader):
            images = images.cuda()
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
            # print(acc1_s)
            losses.update(loss.item(), images.size(0))
            s_top1.update(acc1_s[0], images.size(0))
            t_top1.update(acc1_t[0], images.size(0))

        print(' * S_Acc@1 {s_top1.avg:.3f} T_Acc@1 {t_top1.avg:.3f}'.format(s_top1=s_top1, t_top1=t_top1))

    return s_top1.avg


def main(args):
    # dataset preparation
    classnames = None
    template = None
    val_dataset = None
    if 'ILSVRC' in args.data:
        classnames = imagenet_classnames
        template = openai_imagenet_template
        val_dataset = datasets.ImageFolder(os.path.join(args.data, 'val'),
                                           transform=transforms.Compose([
                                               transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                                    (0.26862954, 0.26130258, 0.27577711)),
                                           ]))
    elif 'caltech256' in args.data:
        classnames = caltech256_classnames
        template = openai_imagenet_template
        val_dataset = datasets.ImageFolder(os.path.join(args.data, 'val'),
                                           transform=transforms.Compose([
                                               transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                                    (0.26862954, 0.26130258, 0.27577711)),
                                           ]))
    elif 'cifar' in args.data:
        classnames = cifar100_classes
        template = openai_imagenet_template
        val_dataset = datasets.CIFAR100(args.data,
                                        train=False,
                                        download=True,
                                        transform=transforms.Compose([
                                            transforms.Resize(224),
                                            # transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                                 (0.26862954, 0.26130258, 0.27577711))
                                        ]))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,  # shuffle for visualization
        num_workers=4)

    clip_model, _ = clip.load('RN50', download_root='./clip/weights/', jit=False)
    classifier = zero_shot_classifier(clip_model, classnames, template).float()

    model = ClipDistillerv7(student=models.__dict__[args.student_arch],
                            teacher=clip_model,
                            classifier=classifier,
                            dim=args.dim,
                            t=args.temp,
                            mlp=args.student_mlp,
                            temp=args.distill_t,
                            m=args.momen,
                            dist=args.distributed)
    # print(args.student_mlp)

    # load checkpoint
    checkpoint = torch.load(args.pretrained)
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        if 'queue' in k or 'queue_ptr' in k:
            del state_dict[k]
        else:
            state_dict[k[len("module."):]] = state_dict[k]

    # print(state_dict.keys())
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.cuda()

    top1 = validate(val_loader, model, soft_cross_entropy, classifier)


if __name__ == '__main__':
    main(parse_opt())

