import argparse
import random
import math
import time
import numpy as np
import torch
import json
from scipy import stats
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from itertools import combinations
from functools import cmp_to_key

from nac_engine import NAC, train_epoch, train_baseline_epoch, evaluate
from noisy_set.genotypes import PRIMITIVES
# from meta_net import MLP
from noisy_set.utils import NAC_DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Args for zero-cost NAS')
parser.add_argument('--seed', type=int, default=301, help='random seed')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=8)  # 要调整，64的时候效果更好？
parser.add_argument('--nac_lr', type=float, default=0.0001)  # 要调整
parser.add_argument('--steps', type=int, default=4, help='number of nodes of a cell')
# parser.add_argument('--meta_net_hidden_size', type=int, default=100)
# parser.add_argument('--meta_net_num_layers', type=int, default=1)
# parser.add_argument('--meta_lr', type=float, default=1e-5)
# parser.add_argument('--meta_weight_decay', type=float, default=0.)
# parser.add_argument('--meta_interval', type=int, default=1)
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
            if mae < 50:
                clean_set.append((arch, mae))

    random.shuffle(clean_set)
    return clean_set


def load_clean_set2(root, epoch):
    # 加载额外clean set
    clean_set = []
    for filename in ["pred_clean1.json", "pred_clean2.json", "pred_clean4.json", "pred_clean5.json",]:
        dir = root + '08/trial3/' + filename

        with open(dir, "r") as f:
            arch_pairs = json.load(f)
        for arch_pair in arch_pairs:
            arch = arch_pair['arch']
            info = arch_pair['info'][:epoch + 1]
            mae = sorted(info, key=lambda x: x[0])[0][0]
            if mae < 50:
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
            info = arch_pair['info'][-1]
            mae = info[0]
            if mae < 50:
                noisy_set.append((arch, mae))

    random.shuffle(noisy_set)
    return noisy_set


def generate_pairs(data):
    # data: [(arch, loss)]
    pairs = []
    data = sorted(data, key=lambda x: x[1])
    for i in range(len(data) - 1):
        for j in range(i + 1, len(data)):
            pairs.append((data[i][0], data[j][0], 1))
            pairs.append((data[j][0], data[i][0], 0))

    return pairs


def sample_arch():
    num_ops = len(PRIMITIVES)
    n_nodes = args.steps

    arch = []
    for i in range(n_nodes):
        if i == 0:
            ops = np.random.choice(range(num_ops), 1)
            nodes = np.random.choice(range(i + 1), 1)
            arch.extend([(nodes[0], ops[0])])
        else:
            ops = np.random.choice(range(num_ops), 2)  # 两条input edge对应两个op（可以相同）
            nodes = np.random.choice(range(i), 1)  # 只有一条可以选择的边
            # nodes = np.random.choice(range(i + 1), 2, replace=False)
            arch.extend([(nodes[0], ops[0]), (i, ops[1])])

    return arch


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

    # 加载clean set和noisy set
    clean_data_dir = 'clean_set/'
    noisy_data_dir = 'noisy_set/'
    clean_set = load_clean_set(clean_data_dir, 99)
    extra_clean_set = load_clean_set2(clean_data_dir, 99)
    # clean_60 = load_clean_set(clean_data_dir, 0)
    # clean_30 = load_clean_set(clean_data_dir, 20)
    # clean_10 = load_clean_set(clean_data_dir, 0)
    noisy_set = load_noisy_set(clean_data_dir)
    # clean_set = sorted(clean_set, key=lambda x: x[-1])
    for pair in clean_set:
        print(pair)
    print('='*30)
    # for pair in extra_clean_set:
    #     print(pair)
    print(len(clean_set))
    # print(len(extra_clean_set))
    print(len(noisy_set))


    # new_noisy_set = []
    # archs, _ = zip(*clean_set)
    # for i, (arch, mae) in enumerate(noisy_set):
    #     if arch not in archs:
    #         new_noisy_set.append((arch, mae))
    # print(len(new_noisy_set))
    # noisy_set = new_noisy_set


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

    valid_set = []
    new_clean_set = []
    for i, (arch, mae) in enumerate(clean_set):
        small_gap = 0
        for j, (arch2, mae2) in enumerate(valid_set):
            if abs(mae - mae2) < 0.08:
                small_gap = 1
                break
        if small_gap == 0:
            valid_set.append((arch, mae))
        else:
            new_clean_set.append((arch, mae))
    print(len(valid_set))
    print(len(new_clean_set))

    # new_clean_set = new_clean_set #+ extra_clean_set
    # train_set = []
    # for i, (arch, mae) in enumerate(new_clean_set):
    #     small_gap = 0
    #     for j, (arch2, mae2) in enumerate(train_set):
    #         if abs(mae - mae2) < 0.05:
    #             small_gap = 1
    #             break
    #     if small_gap == 0:
    #         train_set.append((arch, mae))
    # print(len(train_set))
    # print(len(new_clean_set))

    # train_set = []
    # new_clean_set = []
    # for i, (arch, mae) in enumerate(clean_set):
    #     small_gap = 0
    #     for j, (arch2, mae2) in enumerate(train_set):
    #         if abs(mae - mae2) < 0.05:
    #             small_gap = 1
    #             break
    #     if small_gap == 0:
    #         train_set.append((arch, mae))
    #     else:
    #         new_clean_set.append((arch, mae))
    # print(len(train_set))
    # print(len(new_clean_set))
    #
    # valid_set = []
    # remain_set = []
    # for i, (arch, mae) in enumerate(new_clean_set):
    #     small_gap = 0
    #     for j, (arch2, mae2) in enumerate(valid_set):
    #         if abs(mae - mae2) < 0.05:
    #             small_gap = 1
    #             break
    #     if small_gap == 0:
    #         valid_set.append((arch, mae))
    #     else:
    #         remain_set.append((arch, mae))
    # print(len(valid_set))
    # remain_set.extend(train_set)
    # print(len(remain_set))


    # train_pairs = generate_pairs(train_set)
    valid_pairs = generate_pairs(valid_set)
    noisy_pairs = generate_pairs(noisy_set)
    remain_pairs = generate_pairs(new_clean_set)

    # clean_loader = NAC_DataLoader(train_pairs, args.batch_size)
    valid_loader = NAC_DataLoader(valid_pairs, 1)
    noisy_loader = NAC_DataLoader(noisy_pairs, args.batch_size)
    remain_loader = NAC_DataLoader(remain_pairs, args.batch_size)

    # # build meta-net
    # meta_net = MLP(hidden_size=args.meta_net_hidden_size, num_layers=args.meta_net_num_layers).to(DEVICE)
    # meta_optimizer = optim.Adam(meta_net.parameters(), lr=args.meta_lr, weight_decay=args.meta_weight_decay)

    # build NAC
    nac = NAC(n_nodes=None, n_ops=len(PRIMITIVES), n_layers=4, embedding_dim=128).to(DEVICE)
    criterion = nn.BCELoss()  # 要不要把学习率改成余弦退火？？？或者给GIN加dropout？？？
    nac_optimizer = optim.Adam(nac.parameters(), lr=args.nac_lr, betas=(0.5, 0.999), weight_decay=5e-4)
    # nac_optimizer = optim.SGD(
    #     nac.parameters(),
    #     lr=args.nac_lr,
    #     momentum=0.9,
    #     dampening=0.,
    #     weight_decay=5e-4
    # )

    # his_loss = 100.
    # tolerance = 0
    # for epoch in range(100):  # 训练NAC多少个epochs？好像只能凭经验，因为没有test set
    #     valid_acc, loss = train_baseline_epoch(epoch,
    #                                            noisy_loader,
    #                                            valid_loader,
    #                                            nac,
    #                                            criterion,
    #                                            nac_optimizer)
    #     if loss < his_loss:
    #         tolerance = 0
    #         his_loss = loss
    #         torch.save(nac.state_dict(), "./saved_model/nac1" + ".pth")
    #     else:
    #         tolerance += 1
    #     if tolerance >= 5:
    #         break
    #
    # print('===================================================')
    # nac.load_state_dict(torch.load("./saved_model/nac1" + ".pth"))
    his_loss = 100.
    tolerance = 0
    for epoch in range(100):
        valid_acc, loss = train_baseline_epoch(epoch,
                                               remain_loader,
                                               valid_loader,
                                               nac,
                                               criterion,
                                               nac_optimizer)
        if loss < his_loss:
            tolerance = 0
            his_loss = loss
            torch.save(nac.state_dict(), "./saved_model/nac3" + ".pth")
        else:
            tolerance += 1
        if tolerance >= 5:
            break

    print('===================================================')
    nac.load_state_dict(torch.load("./saved_model/nac3" + ".pth"))
    # his_loss = 100.
    # tolerance = 0
    # for epoch in range(100):
    #     valid_acc, loss = train_baseline_epoch(epoch,
    #                                            clean_loader,
    #                                            valid_loader,
    #                                            nac,
    #                                            criterion,
    #                                            nac_optimizer)
    #     if loss < his_loss:
    #         tolerance = 0
    #         his_loss = loss
    #         torch.save(nac.state_dict(), "./saved_model/nac1" + ".pth")
    #     else:
    #         tolerance += 1
    #     if tolerance >= 5:
    #         break
    # nac.load_state_dict(torch.load("./saved_model/nac1" + ".pth"))

    # random sample and eval
    def compare(arch0, arch1):
        with torch.no_grad():
            nac.eval()
            outputs = nac([arch0], [arch1])
            pred = torch.round(outputs)
        if pred == 0:
            return 1
        else:
            return -1

    archs = []
    for i in range(200000):
        archs.append(sample_arch())
    t1 = time.time()
    # archs应该加上clean和noisy set？
    sorted_archs = sorted(archs, key=cmp_to_key(compare))
    print(f'pred time: {time.time() - t1}')
    print(sorted_archs[:20])
    print(sorted_archs[-1])
    # np.save('./noisy2000', np.array(sorted_archs[:2000]))


if __name__ == '__main__':
    main()
