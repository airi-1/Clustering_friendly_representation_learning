#! /usr/bin/env python

import os
import argparse

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import tqdm.autonotebook as tqdm

import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet

# コマンドラインの引数系
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpus", type=str, default="")
    parser.add_argument("-n", "--num_workers", type=int, default=8)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    return args

def main():
    args = parse()
    # GPUがあるなら、GPUを使って学習する
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 画像の成形と水増し
    tf = [
        # # 画像のトリミングの設定
        # transforms.RandomResizedCrop(size=32,
        #                              scale=(0.2, 1.0),
        #                              ratio=(3 / 4, 4 / 3)),
        # # 画像の明るさをランダムで設定
        # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        # # 画像のグレースケールをランダムで設定
        # transforms.RandomGrayscale(p=0.2),
        # 画像をtensor画像に変換?
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616])
    ]
    # データをロードした後にtfの画像形成処理を行う
    transform = transforms.Compose(tf)


    # データセットの読み込み
    trainset = CIFAR10(root="~/.datasets",
    # train = True(学習データ) False(テストデータ)
                       train=False,
                       download=True,
                       transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset,
    # ミニバッチ学習(読み込んだデータセットから128枚取り出して学習に使う)
                                               batch_size=128,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=args.num_workers)

    # 畳み込みニューラルネットワーク
    low_dim = 128
    net = ResNet18(low_dim=low_dim)
    # L2 norm処理
    norm = Normalize(2)
    # おそらく instance discriminate softmaxの処理?
    npc = NonParametricClassifier(input_dim=low_dim,
                                  output_dim=len(trainset),
                                  tau=1.0,
                                  momentum=0.5)
    loss = Loss(tau2=2.0)
    net, norm = net.to(device), norm.to(device)
    npc, loss = npc.to(device), loss.to(device)
    # 最適化手法"SGD" 学習率を小さくして最初うちへと直線的に収束させる
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=0.03,
                                momentum=0.9,
                                weight_decay=5e-4,
                                nesterov=False,
                                dampening=0)

    # エポック数が節目の1つに達した時に学習率をgammaで減衰
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        [600, 950, 1300, 1650],
                                                        gamma=0.1)
    if torch.cuda.is_available():
        # GPUを並列に使用
        net = torch.nn.DataParallel(net,
                                    device_ids=range(len(
                                        args.gpus.split(","))))
        # 複数の畳み込みアルゴリズムをベンチマークして最速を選択
        torch.backends.cudnn.benchmark = True

    trackers = {n: AverageTracker() for n in ["loss", "loss_id", "loss_fd"]}

    # add -----------------------------------
    # 学習済みモデルの読み込み
    # load_path = './mnt/nvme/internship/idfd_epoch_1999.pth'
    load_path = './drive/MyDrive/idfd_epoch_1999.pth'
    load_weights = torch.load(load_path)
    # 余分なキーが含まれていても無視してくれるらしい
    net.load_state_dict(load_weights,strict=False)

    net.eval()    # 評価モード
    # add -----------------------------------

    # check clustering acc
    # acc, nmi, ari = check_clustering_metrics(npc, train_loader)
    # print("Kmeans ACC, NMI, ARI = {}, {}, {}".format(acc, nmi, ari))

    # # 学習のために2000回繰り返して重みを計算してるらしい
    # with tqdm.trange(2000) as epoch_bar:
    #     for epoch in epoch_bar:

    #

    count = 0

    for batch_idx, (inputs, _,
        indexes) in enumerate(tqdm.tqdm(train_loader)):
    #             optimizer.zero_grad()
        inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
    #             indexes = indexes.to(device, non_blocking=True)
    # CNN backbone処理
        features = norm(net(inputs))
        features_np = features.cpu().detach().numpy().copy()

        if count == 0:
            features_concat = features_np
            count = 1
        else :
            features_concat = np.concatenate([features_concat, features_np])
                # outputs = npc(features, indexes)
    #             loss_id, loss_fd = loss(outputs, features, indexes)
    #             tot_loss = loss_id + loss_fd
    #             tot_loss.backward()
    #             # パラメータの反映
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
    #             # 結果出力 100回やるたびに結果がよくなってることを確認
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
    # Instance discriminate softmaxに溜まっていくデータを使っている?
    # trainFeatures = npc.memory
    # trainFeatures = features
    z = features
    # z = trainFeatures.cpu().numpy()
    # z = trainFeatures.tensor.detach().numpy()
    # 正解データらしいもの
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


# 教師無し分類機_前処理？
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

# 教師無し分類機
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

# 強化
class Normalize(nn.Module):
    def __init__(self, power=2):
        super().__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

# トレーニングされたモデルを返す?
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

# プログラムがコマンドラインから呼ばれた時
if __name__ == "__main__":
    main()

