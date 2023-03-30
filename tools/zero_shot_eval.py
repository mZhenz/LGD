import torch
import clip
from clip.imagenet_zeroshot_data import imagenet_classnames, openai_imagenet_template
from tqdm import tqdm
import numpy as np


def getClassifier(model, classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # print(class_embeddings.shape)
            # class_embedding = class_embeddings.mean(dim=0)
            # class_embedding /= class_embedding.norm()
            # print(class_embedding.shape)
            zeroshot_weights.append(class_embeddings)
        # zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        zeroshot_weights = torch.cat(zeroshot_weights, dim=0).t().float().cuda()
    return zeroshot_weights


def zero_shot_classifier(model, classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(args, model, classifier, dataloader, epoch):
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for _, (images, target) in enumerate(dataloader):
            images = images.cuda()
            target = target.cuda()

            # predict
            image_features = model(images)
            # image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ classifier.float()
            # logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    acc_top1 = (top1 / n) * 100.
    acc_top5 = (top5 / n) * 100.
    print('Test Epoch: {} Top1: {}/{}={:.2f} Top5: {}/{}={:.2f}'.format(epoch, int(top1), int(n), acc_top1, int(top5), int(n), acc_top5))
    return acc_top1, acc_top5


def run_once(args, model, classifier, dataloader, epoch):
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for (images, target) in tqdm(dataloader):
            images = images.cuda()
            target = target.cuda()

            # predict
            # image_features = model(images)
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            # logits = 100. * image_features @ classifier.float()
            logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    acc_top1 = (top1 / n)
    acc_top5 = (top5 / n)
    print('Test Epoch: {} Top1: {}/{}={:.6f} Top5: {}/{}={:.6f}'.format(epoch, int(top1), int(n), acc_top1, int(top5), int(n), acc_top5))
    return acc_top1, acc_top5


def sczero_shot_eval(args, teacher, student, data, epoch):
    classifier = zero_shot_classifier(args, teacher, imagenet_classnames, openai_imagenet_template)
    # classifier = torch.tensor(np.load('./checkpoint/clip_pretrained/imagenet_text_token_1024x1000.npy')).cuda()
    top1, top5 = run(args, student, classifier, data, epoch)
    return top1, top5


if __name__ == '__main__':
    import argparse
    from torchvision import datasets, transforms
    import os

    parser = argparse.ArgumentParser(description='Zero-shot transfer')
    parser.add_argument('--model', type=str, default='RN50',
                        choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16'],
                        help='teacher model name')
    parser.add_argument("--gpu", type=int, default=0,
                        help="Specify a single GPU to run the code on for debugging.")
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--data-root', type=str, default='/dockerdata/ILSVRC/Data/CLS-LOC/')
    args = parser.parse_args()
    device = torch.device('cuda')
    teacher, _ = clip.load(args.model, device=device, download_root='./clip/weights/', jit=False)
    teacher = teacher.to(device)
    teacher.eval()

    # test_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(os.path.join(args.data_root, 'val'),
    #                          transform=transforms.Compose([
    #                              transforms.Resize(256),
    #                              transforms.CenterCrop(224),
    #                              transforms.ToTensor(),
    #                              transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    #                          ])),
    #     batch_size=args.test_batch_size, shuffle=True, num_workers=4)  # shuffle for visualization

    classifier = zero_shot_classifier(teacher, imagenet_classnames, openai_imagenet_template)
    np.save('imagenet_text_token_1024x1000.npy', classifier.numpy())
    print('save done.')
    # top1, top5 = run_once(args, teacher, classifier, test_loader, 0)

