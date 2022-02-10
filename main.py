#! /usr/bin/env python

import os
import argparse

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import tqdm.autonotebook as tqdm

import torch
import time
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet
import torchvision
from torch2trt import torch2trt
import tensorrt as trt

# $B%3%^%s%I%i%$%s$N0z?t7O(B
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpus", type=str, default="")
    parser.add_argument("-n", "--num_workers", type=int, default=8)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    return args

def main():
    args = parse()
    # GPU$B$,$"$k$J$i!"(BGPU$B$r;H$C$F3X=,$9$k(B
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # $B2hA|$N@.7A$H?eA}$7(B
    tf = [
        # # $B2hA|$N%H%j%_%s%0$N@_Dj(B
        # transforms.RandomResizedCrop(size=32,
        #                              scale=(0.2, 1.0),
        #                              ratio=(3 / 4, 4 / 3)),
        # # $B2hA|$NL@$k$5$r%i%s%@%`$G@_Dj(B
        # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        # # $B2hA|$N%0%l!<%9%1!<%k$r%i%s%@%`$G@_Dj(B
        # transforms.RandomGrayscale(p=0.2),
        # $B2hA|$r(Btensor$B2hA|$KJQ49(B?
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616])
    ]
    # $B%G!<%?$r%m!<%I$7$?8e$K(Btf$B$N2hA|7A@.=hM}$r9T$&(B
    transform = transforms.Compose(tf)


    # $B%G!<%?%;%C%H$NFI$_9~$_(B
    trainset = CIFAR10(root="~/.datasets",
    # train = True($B3X=,%G!<%?(B) False($B%F%9%H%G!<%?(B)
                       train=False,
                       download=True,
                       transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset,
    # $B%_%K%P%C%A3X=,(B($BFI$_9~$s$@%G!<%?%;%C%H$+$i(B128$BKg<h$j=P$7$F3X=,$K;H$&(B)
                                               batch_size=128,
    # $B%G!<%?$N=g=x(B($B3X=,(B=True $B?dO@(B=False)
                                               shuffle=False,
                                               pin_memory=True,
                                               num_workers=args.num_workers)

    # $B>v$_9~$_%K%e!<%i%k%M%C%H%o!<%/(B
    low_dim = 128
    net = ResNet18(low_dim=low_dim)
    # L2 norm$B=hM}(B
    norm = Normalize(2)
    # $B$*$=$i$/(B instance discriminate softmax$B$N=hM}(B?
    npc = NonParametricClassifier(input_dim=low_dim,
                                  output_dim=len(trainset),
                                  tau=1.0,
                                  momentum=0.5)
    loss = Loss(tau2=2.0)
    net, norm = net.to(device), norm.to(device)
    npc, loss = npc.to(device), loss.to(device)
    # $B:GE,2=<jK!(B"SGD" $B3X=,N($r>.$5$/$7$F:G=i$&$A$X$HD>@~E*$K<}B+$5$;$k(B
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=0.03,
                                momentum=0.9,
                                weight_decay=5e-4,
                                nesterov=False,
                                dampening=0)

    # $B%(%]%C%/?t$,@aL\$N(B1$B$D$KC#$7$?;~$K3X=,N($r(Bgamma$B$G8:?j(B
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        [600, 950, 1300, 1650],
                                                        gamma=0.1)

    # load_path = './drive/MyDrive/idfd_epoch_1999.pth'
    load_path = './idfd_epoch_1999.pth'
    load_weights = torch.load(load_path,map_location=device)
    # $BM>J,$J%-!<$,4^$^$l$F$$$F$bL5;k$7$F$/$l$k$i$7$$(B
    net.load_state_dict(load_weights,strict=False)

    net.eval()    # $BI>2A%b!<%I(B

    model = torchvision.models.resnet18(pretrained=True).cuda().half().eval()
    data = torch.randn((1, 3, 32, 32)).cuda()
    net = torch2trt(net, [data], max_batch_size=128, fp16_mode=False)

    if torch.cuda.is_available():
        # GPU$B$rJBNs$K;HMQ(B
        net = torch.nn.DataParallel(net,
                                    device_ids=range(len(
                                        args.gpus.split(","))))
        # $BJ#?t$N>v$_9~$_%"%k%4%j%:%`$r%Y%s%A%^!<%/$7$F:GB.$rA*Br(B
        torch.backends.cudnn.benchmark = True

    trackers = {n: AverageTracker() for n in ["loss", "loss_id", "loss_fd"]}

    # add -----------------------------------
    # $B3X=,:Q$_%b%G%k$NFI$_9~$_(B
    # load_path = './mnt/nvme/internship/idfd_epoch_1999.pth'

    # add -----------------------------------

    # check clustering acc
    # acc, nmi, ari = check_clustering_metrics(npc, train_loader)
    # print("Kmeans ACC, NMI, ARI = {}, {}, {}".format(acc, nmi, ari))

    # # $B3X=,$N$?$a$K(B2000$B2s7+$jJV$7$F=E$_$r7W;;$7$F$k$i$7$$(B
    # with tqdm.trange(2000) as epoch_bar:
    #     for epoch in epoch_bar:

    # $BI>2A;~4V7WB,(B start
    start_eval_time = time.perf_counter()
    count = 0

    #add ----------

    for batch_idx, (inputs, _,
        indexes) in enumerate(tqdm.tqdm(train_loader)):
    #             optimizer.zero_grad()
        inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
    #             indexes = indexes.to(device, non_blocking=True)
    # CNN backbone$B=hM}(B
        features = norm(net(inputs))
        features_np = features.cpu().detach().numpy()
        # features_np = features.cpu().detach().numpy().cppy
        if count == 0:
            features_concat = features_np
            count = 1
        else :
            features_concat = np.concatenate([features_concat, features_np])

                # outputs = npc(features, indexes)
    #             loss_id, loss_fd = loss(outputs, features, indexes)
    #             tot_loss = loss_id + loss_fd
    #             tot_loss.backward()
    #             # $B%Q%i%a!<%?$NH?1G(B
    #             optimizer.step()
    #             # track loss
    #             trackers["loss"].add(tot_loss)
    #             trackers["loss_id"].add(loss_id)
    #             trackers["loss_fd"].add(loss_fd)
        # lr_scheduler.step()
    #
    #         # logging
    #         postfix = {name: t.avg() for name, t in trackers.items()}
    #         epoch_bar.set_postfix(**postfix)
    #         for t in trackers.values():
    #             t.reset()
    #
    #
    #         # check clustering acc
    #         if (epoch == 0) or (((epoch + 1) % 100) == 0):
    # acc, nmi, ari = check_clustering_metrics(features, train_loader)
    acc, nmi, ari = check_clustering_metrics(features_concat, train_loader)
                 # acc, nmi, ari = check_clustering_metrics(npc, train_loader)
    #             # $B7k2L=PNO(B 100$B2s$d$k$?$S$K7k2L$,$h$/$J$C$F$k$3$H$r3NG'(B

    # $BI>2A;~4V7WB,(B end
    end_eval_time = time.perf_counter()
    eval_measure = end_eval_time - start_eval_time
    print("eval_measure")
    print(eval_measure)
    print("Epoch:{} Kmeans ACC, NMI, ARI = {}, {}, {}".format(0,acc, nmi, ari))
        # print("Epoch:{} Kmeans ACC, NMI, ARI = {}, {}, {}".format(epoch+1, acc, nmi, ari))

class AverageTracker():
    def __init__(self):
        self.step = 0
        self.cur_avg = 0

    def add(self, value):
        self.cur_avg *= self.step / (self.step + 1)
        self.cur_avg += value / (self.step + 1)
        self.step += 1

    def reset(self):
        self.step = 0
        self.cur_avg = 0

    def avg(self):
        return self.cur_avg.item()


class CIFAR10(datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


def check_clustering_metrics(features, train_loader):
    # Instance discriminate softmax$B$KN/$^$C$F$$$/%G!<%?$r;H$C$F$$$k(B?
    # trainFeatures = npc.memory
    # trainFeatures = features
    z = features
    # z = trainFeatures.cpu().numpy()
    # z = trainFeatures.tensor.detach().numpy()
    # $B@52r%G!<%?$i$7$$$b$N(B
    y = np.array(train_loader.dataset.targets)
    n_clusters = len(np.unique(y))
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z)
    return metrics.acc(y, y_pred), metrics.nmi(y,
                                               y_pred), metrics.ari(y, y_pred)


class metrics:
    ari = adjusted_rand_score
    nmi = normalized_mutual_info_score

    @staticmethod
    def acc(y_true, y_pred):
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        row, col = linear_sum_assignment(w.max() - w)
        return sum([w[i, j] for i, j in zip(row, col)]) * 1.0 / y_pred.size


# $B65;UL5$7J,N`5!(B_$BA0=hM}!)(B
class NonParametricClassifierOP(Function):
    @staticmethod
    def forward(ctx, x, y, memory, params):

        tau = params[0].item()
        out = x.mm(memory.t())
        out.div_(tau)
        ctx.save_for_backward(x, memory, y, params)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, memory, y, params = ctx.saved_tensors
        tau = params[0]
        momentum = params[1]

        grad_output.div_(tau)

        grad_input = grad_output.mm(memory)
        grad_input.resize_as_(x)

        weight_pos = memory.index_select(0, y.view(-1)).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(x.mul(1 - momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)

        return grad_input, None, None, None, None

# $B65;UL5$7J,N`5!(B
class NonParametricClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, tau=1.0, momentum=0.5):
        super(NonParametricClassifier, self).__init__()
        self.register_buffer('params', torch.tensor([tau, momentum]))
        stdv = 1. / np.sqrt(input_dim / 3.)
        self.register_buffer(
            'memory',
            torch.rand(output_dim, input_dim).mul_(2 * stdv).add_(-stdv))

    def forward(self, x, y):
        out = NonParametricClassifierOP.apply(x, y, self.memory, self.params)
        return out

# $B6/2=(B
class Normalize(nn.Module):
    def __init__(self, power=2):
        super().__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

# $B%H%l!<%K%s%0$5$l$?%b%G%k$rJV$9(B?
def ResNet18(low_dim=128):
    net = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], low_dim)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                          stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    return net


class Loss(nn.Module):
    def __init__(self, tau2):
        super().__init__()
        self.tau2 = tau2

    def forward(self, x, ff, y):

        L_id = F.cross_entropy(x, y)

        norm_ff = ff / (ff**2).sum(0, keepdim=True).sqrt()
        coef_mat = torch.mm(norm_ff.t(), norm_ff)
        coef_mat.div_(self.tau2)
        a = torch.arange(coef_mat.size(0), device=coef_mat.device)
        L_fd = F.cross_entropy(coef_mat, a)
        return L_id, L_fd

# $B%W%m%0%i%`$,%3%^%s%I%i%$%s$+$i8F$P$l$?;~(B
if __name__ == "__main__":
    # time$B4X?t(B
    start_all_time = time.perf_counter()
    main()
    end_all_time = time.perf_counter()
    all_measure = end_all_time - start_all_time
    print("all_measure_time:")
    print(all_measure)
