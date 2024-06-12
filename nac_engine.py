import numpy as np
import torch
import torch.nn as nn
from torch.optim.sgd import SGD
from scipy import stats

from gcn_net import GIN, GCN
from noisy_set.genotypes import PRIMITIVES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_baseline_epoch(epoch, data_loader, valid_loader, nap, criterion, optimizer):
    train_loss = []
    nap.train()
    train_dataloader = data_loader
    print(f'epoch num: {epoch}')
    train_dataloader.shuffle()
    for i, (arch, label) in enumerate(train_dataloader.get_iterator()):  # 对每个batch
        label = torch.Tensor(label).to(DEVICE)

        optimizer.zero_grad()
        outputs = nap(arch)
        # predict = scaler.inverse_transform(output)
        loss = criterion(outputs, label)
        train_loss.append(loss.item())

        loss.backward()
        optimizer.step()

    print(f'mse_loss: {np.mean(train_loss)}')
    print('Computing Test Result...')

    # eval
    with torch.no_grad():
        nap = nap.eval()
        valid_loss = []
        for i, (arch, label) in enumerate(valid_loader.get_iterator()):
            label = torch.Tensor(label).to(DEVICE)
            outputs = nap(arch)
            # predict = scaler.inverse_transform(output)
            loss = criterion(outputs, label)
            valid_loss.append(loss.item())

    print(f'valid_mse_loss: {np.mean(valid_loss)}')
    return np.mean(valid_loss)

    # valid_accuracy, valid_loss = evaluate(valid_loader, nac, criterion)
    # return valid_accuracy, valid_loss


def evaluate(val_loader, nac, criterion):
    with torch.no_grad():
        nac.eval()
        valid_loss = []
        preds = []
        for i, (arch, label) in enumerate(val_loader.get_iterator()):
            label = torch.Tensor(label).to(DEVICE)
            output = nac(arch)
            # predict = scaler.inverse_transform(output)
            preds.append(output)
            loss = criterion(output, label)
            valid_loss.append(loss.item())

    return preds, np.mean(valid_loss)

def evaluate2(val_loader, nap, criterion):
    with torch.no_grad():
        nap.eval()
        # valid_loss = []
        preds = []
        labels = []
        for i, (arch, label) in enumerate(val_loader.get_iterator()):
            # label = torch.Tensor(label).to(DEVICE)
            output = nap(arch)
            # predict = scaler.inverse_transform(output)
            preds.append(output.cpu()[0])
            labels.append(label[0])
            # loss = criterion(output, label)
            # valid_loss.append(loss.item())
        # print(labels)
        # print(preds)
        rho = stats.spearmanr(labels, preds)

    return rho


def train_epoch(args, epoch, clean_loader, noisy_loader, valid_loader, nac, meta_net, criterion, optimizer,
                meta_optimizer):
    # MWNet效果可能不好，后面要换更新的方法
    # num_correct = 0
    train_loss = []
    nac.train()
    train_dataloader = noisy_loader
    meta_dataloader = clean_loader
    meta_dataloader_iter = iter(meta_dataloader.get_iterator())
    print(f'epoch num: {epoch}')
    train_dataloader.shuffle()
    for i, (arch0, arch1, label) in enumerate(train_dataloader.get_iterator()):  # 对每个batch

        # arch0 = torch.Tensor(arch0).to(DEVICE)
        # arch1 = torch.Tensor(arch1).to(DEVICE)
        label = torch.Tensor(label).to(DEVICE)
        if (i + 1) % args.meta_interval == 0:  # meta_interval暂时设为1
            pseudo_net = NAC(n_nodes=None, n_ops=len(PRIMITIVES), n_layers=2, embedding_dim=128)
            pseudo_net.load_state_dict(nac.state_dict())
            pseudo_net.train()

            pseudo_outputs = pseudo_net(arch0, arch1)
            pseudo_loss_vector = criterion(pseudo_outputs, label)  # shape=[b, 1]???
            pseudo_loss_vector_reshape = torch.reshape(pseudo_loss_vector, (-1, 1))  # 可能没必要
            pseudo_weight = meta_net(pseudo_loss_vector_reshape.data)
            pseudo_loss = torch.mean(pseudo_weight * pseudo_loss_vector_reshape)

            pseudo_grads = torch.autograd.grad(pseudo_loss, pseudo_net.parameters(), create_graph=True)

            # 为什么要改写optimizer?
            pseudo_optimizer = MetaSGD(pseudo_net, pseudo_net.parameters(), lr=args.nac_lr)
            pseudo_optimizer.load_state_dict(optimizer.state_dict())
            pseudo_optimizer.meta_step(pseudo_grads)  # 在更新后的pseudo net基础上计算meta_net的梯度?

            del pseudo_grads

            try:
                meta_arch0, meta_arch1, meta_labels = next(meta_dataloader_iter)
            except StopIteration:
                meta_dataloader_iter = iter(meta_dataloader.get_iterator())
                meta_arch0, meta_arch1, meta_labels = next(meta_dataloader_iter)

            meta_labels = torch.Tensor(meta_labels).to(DEVICE)
            meta_outputs = pseudo_net(meta_arch0, meta_arch1)
            meta_loss = criterion(meta_outputs, meta_labels)

            meta_optimizer.zero_grad()
            meta_loss.backward()
            meta_optimizer.step()

        outputs = nac(arch0, arch1)  # 在原来的net上，使用新的meta_net，计算梯度
        loss_vector = criterion(outputs, label)
        loss_vector_reshape = torch.reshape(loss_vector, (-1, 1))

        with torch.no_grad():
            weight = meta_net(loss_vector_reshape)

        loss = torch.mean(weight * loss_vector_reshape)
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'train_loss: {np.mean(train_loss)}')
    print('Computing Test Result...')
    valid_accuracy, valid_loss = evaluate(valid_loader, nac, criterion)

    #     optimizer.zero_grad()
    #     outputs = nac(arch0, arch1)
    #     loss = criterion(outputs, label)
    #     loss.backward()
    #     optimizer.step()
    #
    #     train_loss.append(loss.item())
    #
    #     pred = outputs.argmax(dim=1)
    #     num_correct += torch.eq(pred, label).sum().float().item()
    # accuracy = num_correct / len(clean_loader)

    # return accuracy, np.mean(train_loss)

    return valid_accuracy, valid_loss


def geno_to_adj(arch):
    # arch.shape = [7, 2]
    # 输出邻接矩阵，和节点特征
    # 这里的邻接矩阵对应op为顶点的DAG，和Darts相反
    # GCN处理无向图，这里DAG是有向图，所以需要改改？？？参考Wei Wen的文章
    node_num = len(arch) + 2  # 加上一个input和一个output节点
    adj = np.zeros((node_num, node_num))
    ops = [len(PRIMITIVES)]
    for i in range(len(arch)):
        connect, op = arch[i]
        ops.append(arch[i][1])
        if connect == 0 or connect == 1:
            adj[connect][i + 1] = 1
        else:
            adj[(connect - 2) * 2 + 2][i + 1] = 1
            adj[(connect - 2) * 2 + 3][i + 1] = 1
    adj[-3][-1] = 1
    adj[-2][-1] = 1  # output
    ops.append(len(PRIMITIVES) + 1)

    return adj, ops


class NAC(nn.Module):
    def __init__(self, n_nodes, n_ops, n_layers=2, ratio=2, embedding_dim=128):
        # 后面要参考下Wei Wen文章的GCN实现
        super(NAC, self).__init__()
        self.n_nodes = n_nodes
        self.n_ops = n_ops

        # +2用于表示input和output node
        self.embedding = nn.Embedding(self.n_ops + 2, embedding_dim=embedding_dim).to(DEVICE)
        self.gcn = GIN(n_layers=n_layers, in_features=embedding_dim,
                       hidden=embedding_dim, num_classes=embedding_dim).to(DEVICE)

        self.fc = nn.Linear(embedding_dim * ratio, 1, bias=True).to(DEVICE)  # f_out=1  ratio是啥意思？

    def forward(self, arch0, arch1):
        # 先将数组编码改成邻接矩阵编码
        # arch0.shape = [batch_size, 7, 2]

        b_adj0, b_adj1, b_ops0, b_ops1 = [], [], [], []
        for i in range(len(arch0)):
            adj0, ops0 = geno_to_adj(arch0[i])
            adj1, ops1 = geno_to_adj(arch1[i])
            b_adj0.append(adj0)
            b_adj1.append(adj1)
            b_ops0.append(ops0)
            b_ops1.append(ops1)

        b_adj0 = torch.Tensor(b_adj0).to(DEVICE)
        b_adj1 = torch.Tensor(b_adj1).to(DEVICE)
        b_ops0 = torch.LongTensor(b_ops0).to(DEVICE)
        b_ops1 = torch.LongTensor(b_ops1).to(DEVICE)
        feature = torch.cat([self.extract_features((b_adj0, b_ops0)), self.extract_features((b_adj1, b_ops1))], dim=1)

        score = self.fc(feature).view(-1)

        probility = torch.sigmoid(score)

        return probility

    def extract_features(self, arch):
        # 分别输入邻接矩阵和operation？
        if len(arch) == 2:
            matrix, op = arch
            return self._extract(matrix, op)
        else:
            print('error')
        # else:
        #     nor_mat, nor_ops, red_mat, red_ops = arch  # 这个输入是啥意思？
        #     nor_feature = self._extract(nor_mat, nor_ops)
        #     red_feature = self._extract(red_mat, red_ops)
        #     return torch.cat([nor_feature, red_feature], dim=1)

    def _extract(self, matrix, ops):
        # 这里ops是序号编码
        ops = self.embedding(ops)
        feature = self.gcn(ops, matrix).mean(dim=1, keepdim=False)  # shape=[b, nodes, dim] pooling
        return feature


class NAP(nn.Module):
    def __init__(self, n_nodes, n_ops, n_layers=2, embedding_dim=128):
        super(NAP, self).__init__()
        self.n_nodes = n_nodes
        self.n_ops = n_ops

        # +2用于表示input和output node
        self.embedding = nn.Embedding(self.n_ops + 2, embedding_dim=embedding_dim).to(DEVICE)
        self.gcn = GCN(n_layers=n_layers, in_features=embedding_dim,
                       hidden=embedding_dim, num_classes=embedding_dim).to(DEVICE)

        self.fc = nn.Linear(embedding_dim, 1, bias=True).to(DEVICE)  # f_out=1

    def forward(self, arch0):
        # 先将数组编码改成邻接矩阵编码
        # arch0.shape = [batch_size, 7, 2]
        b_adj0, b_adj1, b_ops0, b_ops1 = [], [], [], []
        for i in range(len(arch0)):
            adj0, ops0 = geno_to_adj(arch0[i])
            b_adj0.append(adj0)
            b_ops0.append(ops0)

        b_adj0 = torch.Tensor(b_adj0).to(DEVICE)
        b_ops0 = torch.LongTensor(b_ops0).to(DEVICE)
        feature = self.extract_features((b_adj0, b_ops0))
        score = self.fc(feature).view(-1)  # 参考一下别人的实现

        return score

    def extract_features(self, arch):
        # 分别输入邻接矩阵和operation？
        if len(arch) == 2:
            matrix, op = arch
            return self._extract(matrix, op)
        else:
            print('error')

    def _extract(self, matrix, ops):
        # 这里ops是序号编码
        ops = self.embedding(ops)
        feature = self.gcn(ops, matrix).mean(dim=1, keepdim=False)  # shape=[b, nodes, dim] pooling
        return feature


if __name__ == '__main__':
    arch = [[0, 1], [0, 2], [1, 0], [0, 0], [2, 2], [2, 5], [3, 0]]
    adj, ops = geno_to_adj(arch)
    print(adj)
    print(ops)
