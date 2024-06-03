import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import concat_all_gather


class ClipDistiller(nn.Module):
    """
    Adapted from MoCo, He, Kaiming, et al. "Momentum contrast for unsupervised visual representation learning."
    """

    def __init__(self,
                 student,
                 teacher,
                 classifier,
                 dim=1024,
                 t=0.07,
                 mlp=True,
                 temp=1e-4,
                 m=0.99,
                 dist=True):
        """
        dim:        feature dimension (default: 128)
        K:          queue size
        t:          temperature for student encoder
        temp:       distillation temperature
        base_width: width of the base network
        swav_mlp:   MLP length for SWAV resnet, default=None
        """
        super(ClipDistiller, self).__init__()

        # self.K = K
        self.t = t
        self.temp = temp
        self.dim = dim
        self.dist = dist
        self.m = m

        # create the Teacher/Student encoders
        # num_classes is the output fc dimension
        self.student = student(num_classes=dim)
        self.teacher = teacher
        self.classifier = classifier

        if mlp:
            dim_mlp = self.student.fc.weight.shape[1]
            self.student.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.student.fc)

        # not update by gradient
        for param_k in self.teacher.parameters():
            param_k.requires_grad = False

        # create the queue
        self.C = self.classifier.shape[1]
        # print(self.C)
        self.register_buffer("queue", torch.randn(dim, self.C))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(self.C, dtype=torch.long))

    # queue updation
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, concat=True):
        # gather keys before updating queue in distributed mode
        if concat:
            keys = concat_all_gather(keys)  # BxD

        # print(keys.shape)
        score = torch.einsum('nc,ck->nk', [keys, self.classifier.clone().detach()])  # BxC
        # print(score.shape)
        dummy_cls_results = torch.argmax(score, dim=1)  # B
        # print(dummy_cls_results.shape, dummy_cls_results)
        # print(self.queue_ptr)
        # print(dummy_cls_results)
        for i in range(self.C):
            flag = int(self.queue_ptr[i])
            if i not in dummy_cls_results:
                continue
            z_c = keys[dummy_cls_results == i]
            # print(z_c.shape)
            z_c = torch.mean(z_c, dim=0)
            # z_c = F.normalize(torch.mean(z_c, dim=0), dim=0)
            if flag:
                # print(z_c.shape, self.queue[:,i].shape)
                self.queue[:, i] = self.m * self.queue[:, i] + (1 - self.m) * z_c
            else:
                self.queue[:, i] = z_c
                self.queue_ptr[i] = 1

    def forward(self, image, inference=False):
        """
        Input:
            image: a batch of images
        Output:
            student logits, teacher logits
        """
        # compute student features
        s_emb = self.student(image)  # BxD
        s_emb = F.normalize(s_emb, dim=1)

        # compute teacher features
        with torch.no_grad():  # no gradient to keys
            t_emb = self.teacher.encode_image(image).float()  # keys: (HW+1)xBxD
            # t_emb = F.normalize(t_emb[0], dim=1)  # BxD
            t_emb = F.normalize(t_emb[0], dim=1)  # BxD

        # compute image-image logits
        logit_stu_img = torch.einsum('nc,ck->nk', [s_emb, self.queue.clone().detach()])  # BxK
        logit_tea_img = torch.einsum('nc,ck->nk', [t_emb, self.queue.clone().detach()])  # BxK

        logit_s_p = torch.einsum('nc,nc->n', [s_emb, t_emb]).unsqueeze(-1)  # Bx1
        logit_t_p = torch.einsum('nc,nc->n', [t_emb, t_emb]).unsqueeze(-1)  # Bx1

        logit_stu_img = torch.cat([logit_s_p, logit_stu_img], dim=1)  # Bx(K+1)
        logit_tea_img = torch.cat([logit_t_p, logit_tea_img], dim=1)  # Bx(K+1)

        logit_stu_img /= self.t
        logit_tea_img = F.softmax(logit_tea_img / self.temp, dim=1)

        # compute image-text logits
        logit_stu_text = torch.einsum('nc,ck->nk', [s_emb, self.classifier.clone().detach()])  # BxC
        logit_tea_text = torch.einsum('nc,ck->nk', [t_emb, self.classifier.clone().detach()])  # BxC

        logit_stu_text /= self.t
        logit_tea_text = F.softmax(logit_tea_text / self.temp, dim=1)

        # de-queue and en-queue
        if not inference:
            self._dequeue_and_enqueue(t_emb, concat=self.dist)

        return logit_stu_img, logit_tea_img, logit_stu_text, logit_tea_text, s_emb, t_emb


if __name__ == '__main__':
    import seed.models as models
    import clip

    student = models.__dict__['resnet18']
    clip_model, _ = clip.load('RN50', download_root='/cfs/cfs-31b43a0b8/personal/mingzhenzhu/CLIP/weights/', jit=False)
    classifier = torch.randn(1024, 1000).cuda()
    model = ClipDistillerv9(student=student,
                            teacher=clip_model,
                            classifier=classifier,
                            dim=1024,
                            K=60,
                            t=0.2,
                            mlp=True,
                            temp=0.01,
                            dist=False).cuda()

    # print(model)

    for i in range(10):
        print('========> This is run {}'.format(i))
        dummy_input = torch.randn(1024, 3, 224, 224).cuda()
        _, _, _, _ = model(dummy_input)


