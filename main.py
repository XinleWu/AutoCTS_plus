import argparse
import random
import math
import numpy as np
import torch
import json
from scipy import stats
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from itertools import combinations

from nac_engine import NAC, train_epoch, train_baseline_epoch, evaluate
from clean_set.genotypes import PRIMITIVES
from meta_net import MLP
from noisy_set.utils import NAC_DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Args for zero-cost NAS')
parser.add_argument('--seed', type=int, default=3, help='random seed')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--nac_lr', type=float, default=0.0001)  # 0.001+adam or 0.0001+adam
parser.add_argument('--meta_net_hidden_size', type=int, default=100)
parser.add_argument('--meta_net_num_layers', type=int, default=1)
parser.add_argument('--meta_lr', type=float, default=1e-5)
parser.add_argument('--meta_weight_decay', type=float, default=0.)
parser.add_argument('--meta_interval', type=int, default=1)
args = parser.parse_args()


def load_clean_set(root, epoch):
    # epoch1:100
    clean_set = []
    for i in range(1, 4):
        if i == 1:
            dir = root + '08/trial3/' + 'clean_trans.json'
        else:
            dir = root + '08/trial3/' + f'clean_trans{i}.json'

        with open(dir, "r") as f:
            arch_pairs = json.load(f)

        for arch_pair in arch_pairs:
            arch = arch_pair['arch']
            info = arch_pair['info'][:epoch+1]
            mae = sorted(info, key=lambda x: x[0])[0][0]
            if mae < 120:
                clean_set.append((arch, mae))

    random.shuffle(clean_set)
    return clean_set


def load_noisy_set(root):
    noisy_set = []
    for i in range(2, 4):
        dir = root + '08/trial3/' + f'noisy_trans{i}.json'

        with open(dir, "r") as f:
            arch_pairs = json.load(f)
        for arch_pair in arch_pairs:
            arch = arch_pair['arch']
            info = arch_pair['info'][0]
            mae = info[0]
            if mae < 100:
                noisy_set.append((arch, mae))

    random.shuffle(noisy_set)
    return noisy_set[:500]


def generate_pairs(data):
    # data: [(arch, loss)]
    pairs = []
    data = sorted(data, key=lambda x: x[1])
    for i in range(len(data) - 1):
        for j in range(i + 1, len(data)):  # better than m(m-1)/2
            pairs.append((data[i][0], data[j][0], 1))
            pairs.append((data[j][0], data[i][0], 0))

    return pairs


def main():

    print(args)
    if args.cuda and not torch.cuda.is_available():
        args.cuda = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.cuda:
        torch.backends.cudnn.deterministic = True

    # 生成clean set的时候，要观察下，只训练30 epochs， 50 epochs和训练100 epochs得到的排序结果的差距。(to do!)
    # 还要考虑设置更小的hidden_dim? 尽量减少computation cost

    # 加载clean set和noisy set
    clean_data_dir = 'clean_set/'
    noisy_data_dir = 'noisy_set/'
    clean_set = load_clean_set(clean_data_dir, 99)
    clean_60 = load_clean_set(clean_data_dir, 0)
    # clean_30 = load_clean_set(clean_data_dir, 20)
    # clean_10 = load_clean_set(clean_data_dir, 0)
    noisy_set = load_noisy_set(clean_data_dir)
    for pair in clean_set:
        print(pair)
    print(len(clean_set))
    print(len(noisy_set))


    new_noisy_set = []
    archs, _ = zip(*clean_set)
    for i, (arch, mae) in enumerate(noisy_set):
        if arch not in archs:
            new_noisy_set.append((arch, mae))
    print(len(new_noisy_set))
    noisy_set = new_noisy_set


    # print(np.array(list(zip(*clean_set))[1]))
    # print(np.array(list(zip(*clean_60))[1]))
    # print(np.argsort(np.array(list(zip(*clean_set))[1])))
    # print(np.argsort(np.array(list(zip(*clean_60))[1])))

    # KTau_60, _ = stats.kendalltau(np.argsort(np.array(list(zip(*clean_set))[1])),
    #                               np.argsort(np.array(list(zip(*clean_60))[1])))
    # print(f'KTau 60: {KTau_60}')

    # ground_truth = []
    # clean_set = sorted(clean_set, key=lambda x: x[1])
    # for i in range(len(clean_set) - 1):
    #     for j in range(i + 1, len(clean_set)):
    #         ground_truth.append((clean_set[i][0], clean_set[j][0], 1))
    #
    # ground_60 = []
    # clean_60 = sorted(clean_60, key=lambda x: x[1])
    # for i in range(len(clean_60) - 1):
    #     for j in range(i + 1, len(clean_60)):
    #         ground_60.append((clean_60[i][0], clean_60[j][0], 1))
    #
    # num = 0
    # for i in range(len(ground_truth)):
    #     if ground_truth[i] in ground_60:
    #         num += 1
    #         # print(ground_truth[i])
    # print(num / len(ground_truth))

    train_set = []
    new_clean_set = []
    for i, (arch, mae) in enumerate(clean_set):
        small_gap = 0
        for j, (arch2, mae2) in enumerate(train_set):
            if abs(mae - mae2) < 0.1:
                small_gap = 1
                break
        if small_gap == 0:
            train_set.append((arch, mae))
        else:
            new_clean_set.append((arch, mae))
    print(len(train_set))
    print(len(new_clean_set))

    valid_set = []
    for i, (arch, mae) in enumerate(new_clean_set):
        small_gap = 0
        for j, (arch2, mae2) in enumerate(valid_set):
            if abs(mae - mae2) < 0.1:
                small_gap = 1
                break
        if small_gap == 0:
            valid_set.append((arch, mae))
    print(len(valid_set))



    # random.shuffle(clean_set)
    # train_set = clean_set[:20]  # 总共156个，暂时80个用于训练，几乎快到搜索空间的0.1%了
    # valid_set = clean_set[30:]  # 设为20的时候，正常了？和GCN的hidden dim也没关系啊

    train_pairs = generate_pairs(train_set)
    valid_pairs = generate_pairs(valid_set)
    # noisy_set = random.sample(list(noisy_set), 110)
    noisy_pairs = generate_pairs(noisy_set)

    clean_loader = NAC_DataLoader(train_pairs, args.batch_size)
    valid_loader = NAC_DataLoader(valid_pairs, 1)
    noisy_loader = NAC_DataLoader(noisy_pairs, args.batch_size)


    # clean_pairs = generate_pairs(new_clean_set)
    # noisy_pairs = generate_pairs(noisy_set)  # 保留前500试试
    # random.shuffle(clean_pairs)
    # random.shuffle(noisy_pairs)
    # print(len(clean_pairs))
    # print(len(noisy_pairs))
    #
    # # 得将clean分成train val和test？
    # clean_loader = NAC_DataLoader(clean_pairs[:1500], args.batch_size)
    # valid_loader = NAC_DataLoader(clean_pairs[1500:], args.batch_size)
    # noisy_loader = NAC_DataLoader(noisy_pairs[:200000], args.batch_size)
    # noisy_loader2 = NAC_DataLoader(noisy_pairs[200000:], args.batch_size)
    # mix_loader = NAC_DataLoader(clean_pairs[:1500] + noisy_pairs, args.batch_size)

    # build meta-net
    meta_net = MLP(hidden_size=args.meta_net_hidden_size, num_layers=args.meta_net_num_layers).to(DEVICE)
    meta_optimizer = optim.Adam(meta_net.parameters(), lr=args.meta_lr, weight_decay=args.meta_weight_decay)

    # build NAC
    nac = NAC(n_nodes=None, n_ops=len(PRIMITIVES), n_layers=2, embedding_dim=128).to(DEVICE)
    criterion = nn.BCELoss()
    nac_optimizer = optim.Adam(nac.parameters(), lr=args.nac_lr, betas=(0.5, 0.999), weight_decay=5e-4)
    # nac_optimizer = optim.SGD(
    #     nac.parameters(),
    #     lr=args.nac_lr,
    #     momentum=0.9,
    #     dampening=0.,
    #     weight_decay=5e-4
    # )

    for epoch in range(0):  # 训练NAC多少个epochs？好像只能凭经验，因为没有test set
        # valid_acc, loss = train_epoch(args,  # 要把clean set加入到noisy set？
        #                               epoch,
        #                               clean_loader,
        #                               noisy_loader,
        #                               valid_loader,
        #                               nac,
        #                               meta_net,
        #                               criterion,
        #                               nac_optimizer,
        #                               meta_optimizer)  # train NAC
        # print(f'valid_acc: {valid_acc}, valid_loss: {loss}')

        valid_acc, loss = train_baseline_epoch(epoch,
                                               noisy_loader,
                                               valid_loader,
                                               nac,
                                               criterion,
                                               nac_optimizer)

    # valid_acc, loss = evaluate(valid_loader, nac, criterion)
    # print(f'acc: {valid_acc}, loss: {loss}')
    #
    print('===================================================')
    for epoch in range(100):
        valid_acc, loss = train_baseline_epoch(epoch,
                                               clean_loader,
                                               valid_loader,
                                               nac,
                                               criterion,
                                               nac_optimizer)

        # print(f'valid_acc: {valid_acc}, valid_loss: {loss}')
        # # 如果有test set，就在这里加上eval
        # val_acc = evaluate(val_set)


if __name__ == '__main__':
    main()
