import enum
from operator import getitem, itemgetter, length_hint
from numpy.core.defchararray import center
from numpy.lib.function_base import interp
# import pandas as pd
import torch
from torch.autograd import backward
from torch.functional import norm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import ot

import os
import sys
import argparse
import functools
from tqdm import tqdm
import pdb
import copy
import time

from loaders.domainnet import build_dataset, build_dataset_tiny
from loaders.office_home import build_dataset_officehome
from loaders.visda import build_dataset_visda
from utils.utils import AverageMeter, set_seed, weights_init, print_options, AllMeters
from utils.lr_schedule import InvLr
from utils.ema import ModelEMA
from model.basenet import AlexNetBase, Predictor, Predictor_cos_feat, Predictor_deep, PredictorTSA
from model.resnet import resnet34
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from ot_utils import Prototype


def get_args():
    parser = argparse.ArgumentParser(description='Domain label estimation')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr_cent', default=0.5, type=float)
    parser.add_argument('--bs', default=24, type=int)
    parser.add_argument('--bs_unl_multi', default=2, type=int)
    parser.add_argument('--n_workers', default=6, type=int)
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--source', type=str, default='painting')
    parser.add_argument('--target', type=str, default='real')
    parser.add_argument('--seed', type=int, default=12345, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--arch', type=str, default='resnet',
                        help='which network to use')
    parser.add_argument('--data_root', type=str, default='/data/zizheng/domainnet')
    parser.add_argument('--unl_transform', type=str, default='fixmatch')
    parser.add_argument('--labeled_transform', type=str, default='labeled')
    parser.add_argument('--num', type=int, default=3,
                        help='number of labeled examples in the target')
    parser.add_argument('--num_classes', type=int, default=126)
    parser.add_argument('--dataset', type=str, default='multi',
                        choices=['multi', 'office', 'office_home', 'visda'],
                        help='the name of dataset')
    parser.add_argument('--base_path', type=str, default='./data/txt/multi/')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--threshold2', default=0.9, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--T', type=float, default=0.05, metavar='T',
                        help='temperature (default: 0.05)')
    parser.add_argument('--T2', type=float, default=1, metavar='T',
                        help='temperature (default: 0.05)')
    parser.add_argument('--mix_T', type=float, default=0.07, metavar='T',
                        help='temperature (default: 0.05)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lambda_b', type=float, default=0.5)
    parser.add_argument('--lambda_u', type=float, default=1)
    parser.add_argument('--lambda_mcc', type=float, default=1)
    parser.add_argument('--lambda_entmax', type=float, default=0.1)
    parser.add_argument('--lambda_entmax2', type=float, default=0.1)
    parser.add_argument('--lambda_scatter', type=float, default=0)
    parser.add_argument('--lambda_align', type=float, default=0)
    parser.add_argument('--lambda_ot', type=float, default=1)
    parser.add_argument('--lambda_mme', type=float, default=0)
    parser.add_argument('--lambda_g', type=float, default=0)
    parser.add_argument('--eta_mme', type=float, default=0)
    parser.add_argument('--w_kld', type=float, default=0)
    parser.add_argument('--log_dir', type=str, default='./logs/domainnet/contras_cls/debug')
    parser.add_argument('--G_path', type=str, default='')
    parser.add_argument('--kld', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--test_interval', type=float, default=500)
    parser.add_argument('--print_interval', type=float, default=50)
    parser.add_argument('--num_steps', type=int, default=50000)
    parser.add_argument('--momentum', type=float, default=.9)
    parser.add_argument('--prop', type=float, default=.1)
    parser.add_argument('--warm_steps', type=int, default=100)
    args = parser.parse_args()
    return args


def cross_mcc(p1, p2):
    global args
    N, C = p1.shape
    cov = p1.t() @ p2
    loss = (cov - torch.eye(C).cuda()).abs().sum() / C
    return loss


def cross_mcc2(p1, p2):
    global args
    # p1 = p1.detach()
    N, C = p1.shape
    cov = p1.t() @ p2

    cov_norm1 = cov / torch.sum(cov, dim=1, keepdims=True)
    cov_norm2 = cov / torch.sum(cov, dim=0, keepdims=True)
    loss = 0.5 * (torch.sum(cov_norm1) - torch.trace(cov_norm1)) / C \
        + 0.5 * (torch.sum(cov_norm2) - torch.trace(cov_norm2)) / C
    return loss


def cross_mcc_scale(p1, p2):
    '''
    p1 weak, p2 strong
    '''
    global args
    N, C = p1.shape
    cov = p1.t() @ p2.detach()
    loss1 = (cov - torch.eye(C).cuda()).abs().sum() / C

    cov = p1.t().detach() @ p2
    loss2 = (cov - torch.eye(C).cuda()).abs().sum() / C

    loss = 0 * loss1 + 2 * loss2
    return loss


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def cross_mcc_sub(p1, p2):
    global args
    N, C = p1.shape
    cov = p1.t() @ p2
    mask = torch.ones(C).cuda()
    diff = p1.max(dim=0)[0] - p1.min(dim=0)[0]
    loss = off_diagonal(cov).sum() + ((cov.diag() - 1).abs() * (diff > 0.05).float()).sum()
    loss = loss / C
    return loss


def adv_cross_mcc(F1, feat_tu, eta=1.0):
    feat_w, feat_s = feat_tu.chunk(2)
    _, logits_w = F1(feat_w, reverse=True, eta=eta)
    _, logits_s = F1(feat_s, reverse=True, eta=eta)
    prob_w = torch.softmax(logits_w, dim=1)
    prob_s = torch.softmax(logits_s, dim=1)
    loss = -cross_mcc2(prob_w, prob_s)
    return loss


def pairwise_cosine_sim(a, b):
    assert len(a.shape) == 2
    assert a.shape[1] == b.shape[1]
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    mat = a @ b.t()
    return mat


def ot_mapping(M, args):
    '''
    M: (ns, nt)
    '''
    reg1 = 1
    reg2 = 1
    method = 'unbalance'
    ns, nt = M.shape
    a, b = np.ones((ns,)) / ns, np.ones((nt,)) / nt
    if method == 'ent':
        gamma = ot.sinkhorn(a, b, M, reg1)
    elif method == 'unbalance':
        gamma = ot.unbalanced.sinkhorn_stabilized_unbalanced(a, b, M, reg1, reg2)
    elif method == 'emd':
        gamma = ot.emd(a, b, M)
    gamma = torch.from_numpy(gamma).cuda()
    return gamma


def ot_loss(proto_s, feat_tu_w, feat_tu_s, pl):
    global args
    bs = feat_tu_s.shape[0]
    with torch.no_grad():
        M_st_weak = 1 - pairwise_cosine_sim(proto_s.mo_pro, feat_tu_w)  # postive distance
    gamma_st_weak = ot_mapping(M_st_weak.data.cpu().numpy().astype(np.float64), args)
    score_ot, pred_ot = gamma_st_weak.t().max(dim=1)
    Lm = center_loss_cls(proto_s.mo_pro, feat_tu_s, pred_ot, num_classes=args.num_classes)
    return Lm


def ot_loss2(proto_s, feat_tu_w, prob_tu_w, feat_tu_s, prob_tu_s, pl):
    global args
    proto_gt = F.one_hot(torch.arange(args.num_classes)).cuda().float()
    ground_cost = -proto_gt.t() @ torch.log(prob_tu_w).t()
    bs = feat_tu_s.shape[0]
    with torch.no_grad():
        M_st_weak = 1 - pairwise_cosine_sim(proto_s.mo_pro, feat_tu_w)  # postive distance
        M_st_weak += args.lambda_g * ground_cost
    gamma_st_weak = ot_mapping(M_st_weak.data.cpu().numpy().astype(np.float64), args)
    score_ot, pred_ot = gamma_st_weak.t().max(dim=1)
    Lm = center_loss_cls2(proto_s.mo_pro, feat_tu_s, prob_tu_s, pred_ot, lambda_g=args.lambda_g, num_classes=args.num_classes)
    return Lm


def entmax(p):
    p = p.sum(dim=0)
    p /= p.sum()
    loss = (p * p.log()).sum()
    return loss


def scatter(p):
    return (p * p.log()).sum(dim=1).mean()


def adentropy(F1, feat, lamda=0.1, eta=1.0):
    _, out_t1 = F1(feat, reverse=True, eta=eta)
    out_t1 = F.softmax(out_t1, dim=1)
    loss_adent = lamda * torch.mean(torch.sum(out_t1 * (torch.log(out_t1 + 1e-5)), 1))
    return loss_adent


def fix_scatter(F1, feat, temp, lamda=0.1):
    F2 = copy.deepcopy(F1.fc2)
    for param in F2.parameters():
        param.requires_grad = False
    out_t1 = F2(feat) / temp
    p = F.softmax(out_t1, dim=1)
    return lamda * (p * torch.log(p + 1e-5)).sum(dim=1).mean()


def no_grad_F(F1, feat, temp):
    F2 = copy.deepcopy(F1.fc2)
    for param in F2.parameters():
        param.requires_grad = False
    out_t1 = F2(feat) / temp
    return out_t1


def labeled_align(feat_s, feat_t, gt_s, gt_t):
    global args
    B = feat_s.shape[0]
    C = args.num_classes
    # dist = 0.01 * torch.cdist(feat_s, feat_t) ** 2
    dist = -feat_s @ feat_t.t() + 1
    m1 = gt_s.unsqueeze(1).repeat(1, B)
    m2 = gt_t.repeat(B, 1)
    mask = m1.eq(m2)
    if mask.sum() > 0:
        dist = dist * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / mask.float().sum()
    else:
        loss = torch.zeros(1).cuda()
    return loss


def cosine_scale(step, max_v=1, min_v=0):
    global args
    return min_v + 0.5 * (max_v - min_v) * (1 + np.cos(step / args.num_steps * np.pi))


def center_loss_cls(centers, x, labels, num_classes=65):
    classes = torch.arange(num_classes).long().cuda()
    batch_size = x.size(0)
    centers_norm = F.normalize(centers)
    x = F.normalize(x)
    distmat = -x @ centers_norm.t() + 1

    labels = labels.unsqueeze(1).expand(batch_size, num_classes)
    mask = labels.eq(classes.expand(batch_size, num_classes))

    dist = distmat * mask.float()
    loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

    return loss


def center_loss_cls2(centers, x, prob_tu_s, labels, lambda_g, num_classes=65):
    logsoftmax = torch.log(prob_tu_s)
    ce_loss = F.nll_loss(logsoftmax, labels) * lambda_g
    ce_loss += torch.sum(-logsoftmax / logsoftmax.shape[1], dim=1).mean() * lambda_g * 0.1
    # ce_loss = torch.log(prob_tu_s) *

    classes = torch.arange(num_classes).long().cuda()
    batch_size = x.size(0)
    centers_norm = F.normalize(centers)
    x = F.normalize(x)
    distmat = -x @ centers_norm.t() + 1

    labels = labels.unsqueeze(1).expand(batch_size, num_classes)
    mask = labels.eq(classes.expand(batch_size, num_classes))

    dist = distmat * mask.float()
    loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size + ce_loss

    return loss


def proto_mcc(protos, feat_tu_w, feat_tu_s):
    global args
    feat_tu_w = F.normalize(feat_tu_w, dim=1)
    feat_tu_s = F.normalize(feat_tu_s, dim=1)
    logits_w = feat_tu_w @ protos.t()
    logits_s = feat_tu_s @ protos.t()
    proto_prob_w = torch.softmax(logits_w / args.T, dim=1)
    proto_prob_s = torch.softmax(logits_s / args.T, dim=1)
    L_mcc_feat = cross_mcc(proto_prob_s, proto_prob_w)
    L_ent_feat_1 = entmax(proto_prob_w)
    L_ent_feat_2 = entmax(proto_prob_s)
    return L_mcc_feat


@torch.no_grad()
def update_prob_ema(prob_ema, prob_tu_w, m=0.9):
    prob_ema = prob_ema * m + prob_tu_w.mean(dim=0) * (1 - m)
    return prob_ema


@torch.no_grad()
def get_threshold(prob_ema, prob_tu_w, base_threshold):
    prob_norm = prob_ema / prob_ema.max()
    rescaled_threshold = (base_threshold - 0.1) + 0.1 * prob_norm
    # print(rescaled_threshold)
    pl, _ = prob_tu_w.max(dim=1)
    new_thres = rescaled_threshold[pl.long()]
    return new_thres


class CrossEntropyKLD(object):
    def __init__(self, num_class=126, mr_weight_kld=0.1):
        self.num_class = num_class
        self.mr_weight_kld = mr_weight_kld

    def __call__(self, pred, label, mask):
        # valid_reg_num = len(label)
        logsoftmax = F.log_softmax(pred, dim=1)

        kld = torch.sum(-logsoftmax / self.num_class, dim=1)
        ce = (F.cross_entropy(pred, label, reduction='none') * mask).mean()
        kld = (self.mr_weight_kld * kld * mask).mean()

        ce_kld = ce + kld

        return ce_kld


# Training
def train(source_loader, target_loader, target_loader_unl, target_loader_val, target_loader_test,
          net_G, net_F, optimizer, scheduler):
    global args, writer
    net_G.train()
    net_F.train()

    loss_name_list = ['Lx', 'L_fix', 'mask_prop', 'L_mcc', 'L_scatter', 'L_align', 'L_mme']
    train_meters = AllMeters(loss_name_list)

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    target_unl_iter = iter(target_loader_unl)

    criterion = nn.CrossEntropyLoss().cuda()
    best_acc_val, cur_acc_test = 0, 0
    start_time = time.time()

    net_G_ema = ModelEMA(net_G, decay=0.99)
    net_F_ema = ModelEMA(net_F, decay=0.99)
    net_G_ema2 = ModelEMA(net_G, decay=0.995)
    net_F_ema2 = ModelEMA(net_F, decay=0.995)

    proto_s = Prototype(C=args.num_classes, dim=args.inc)
    proto_st = Prototype(C=args.num_classes, dim=args.inc)

    prob_ema = torch.ones(args.num_classes).cuda() / args.num_classes

    CE_KLD = CrossEntropyKLD(num_class=args.num_classes, mr_weight_kld=args.w_kld)
    init_mcc_weight = args.lambda_mcc
    for batch_idx in range(args.num_steps):
        lambda_warm = 1 if batch_idx > args.warm_steps else 0
        # try:
        #     data_batch_source = source_iter.next()
        #     data_batch_target = target_iter.next()
        #     data_batch_unl = target_unl_iter.next()

        # except:
        #     source_iter = iter(source_loader)
        #     target_loader.dataset.shuffle_repeat()
        #     target_iter = iter(target_loader)
        #     target_unl_iter = iter(target_loader_unl)

        #     data_batch_source = source_iter.next()
        #     data_batch_target = target_iter.next()
        #     data_batch_unl = target_unl_iter.next()
        try:
            data_batch_source = source_iter.next()
        except:
            source_iter = iter(source_loader)
            data_batch_source = source_iter.next()

        try:
            data_batch_target = target_iter.next()
        except:
            target_loader.dataset.shuffle_repeat()
            target_iter = iter(target_loader)
            data_batch_target = target_iter.next()

        try:
            data_batch_unl = target_unl_iter.next()
        except:
            target_unl_iter = iter(target_loader_unl)
            data_batch_unl = target_unl_iter.next()

        imgs_s_w = data_batch_source[0].cuda()
        gt_s = data_batch_source[1].cuda()

        imgs_t_w = data_batch_target[0].cuda()
        gt_t = data_batch_target[1].cuda()

        imgs_tu_w, imgs_tu_s = data_batch_unl[0][0].cuda(), data_batch_unl[0][1].cuda()
        gt_tu = data_batch_unl[1].cuda()

        data = torch.cat((imgs_s_w, imgs_t_w), 0)
        target = torch.cat((gt_s, gt_t), 0)
        output = net_G(data)

        feat_t_con, out1 = net_F(output)
        Lx = criterion(out1, target)
        out_s, out_t = out1.chunk(2)
        feat_s, feat_t = feat_t_con.chunk(2)

        # target unl
        feat_tu = net_G(torch.cat((imgs_tu_w, imgs_tu_s), dim=0))
        feat_tu_con, logits_tu = net_F(feat_tu)
        feat_tu_w, feat_tu_s = feat_tu_con.chunk(2)
        logits_tu_w, logits_tu_s = logits_tu.chunk(2)

        # fix match loss
        pseudo_label = torch.softmax(logits_tu_w.detach() * args.T2, dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        consis_mask = max_probs.ge(args.threshold).float()
        if args.kld:
            L_fix = CE_KLD(logits_tu_s, targets_u, consis_mask)
        else:
            L_fix = (F.cross_entropy(logits_tu_s, targets_u, reduction='none') * consis_mask).mean()

        # mcc loss
        prob_tu_w = torch.softmax(logits_tu_w, dim=1)
        prob_tu_s = torch.softmax(logits_tu_s, dim=1)
        # L_mcc = cross_mcc(prob_tu_w, prob_tu_s)
        L_mcc = cross_mcc2(prob_tu_w, prob_tu_s)
        # L_mcc = adv_cross_mcc(net_F, feat_tu)
        L_ent_1 = entmax(prob_tu_w)
        L_ent_2 = entmax(prob_tu_s)

        # source scatter loss
        prob_s = torch.softmax(out_s, dim=1)
        L_scatter = scatter(prob_s)

        # labeled align
        # L_align = labeled_align(feat_s, feat_t, gt_s, gt_t)
        L_align = center_loss_cls(proto_s.mo_pro, feat_t, gt_t, num_classes=args.num_classes)

        # mcc proto
        # L_mcc_proto = proto_mcc(proto_s.mo_pro, feat_tu_w, feat_tu_s)
        # L_mme = adentropy(net_F, feat_tu_w, lamda=args.lambda_mme, eta=args.eta_mme)

        # ot loss
        L_ot = ot_loss(proto_s, feat_tu_w, feat_tu_s, targets_u)
        # L_ot = ot_loss2(proto_s, feat_tu_w, prob_tu_w, feat_tu_s, prob_tu_s, targets_u)
        # L_ot = 0.5 * ot_loss(proto_s, feat_tu_w, feat_tu_s, targets_u) + 0.5 * ot_loss(proto_st, feat_tu_w, feat_tu_s, targets_u)

        # backward
        Loss = Lx + L_scatter * args.lambda_scatter + L_fix * args.lambda_u \
            + L_mcc * args.lambda_mcc + (L_ent_1 + L_ent_2) * args.lambda_mcc * args.lambda_entmax  \
            + lambda_warm * L_ot * args.lambda_ot

        Loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(batch_idx)
        net_G_ema.update(net_G)
        net_F_ema.update(net_F)
        net_G_ema2.update(net_G)
        net_F_ema2.update(net_F)

        proto_s.update(feat_s, gt_s, batch_idx, norm=True)
        prob_ema = update_prob_ema(prob_ema, prob_tu_w)
        proto_st.update(feat_s, gt_s, batch_idx, norm=True)
        proto_st.update(feat_t, gt_t, batch_idx, norm=True)

        loss_name_list = ['Lx', 'L_fix', 'mask_prop', 'L_mcc', 'L_scatter', 'L_align']
        loss_value_list = [Lx.item(), L_fix.item(), consis_mask.sum().item() / consis_mask.shape[0],
                           L_mcc.item(), L_scatter.item(), L_align.item()]
        train_meters.update_list(loss_name_list, loss_value_list)

        if batch_idx % args.print_interval == 0:
            log_str = train_meters.log_str(batch_idx, args)
            print(log_str)
            # print(args.lambda_mcc)

        if (batch_idx + 1) % args.test_interval == 0:
            train_meters.tb_log(writer, batch_idx)
            writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], batch_idx)

            acc_val = test(target_loader_val, net_G_ema.ema, net_F_ema.ema, batch_idx)
            acc1, acc2, acc, per_acc1, per_acc2, per_acc, _ = test_multi(target_loader_test,
                                                                         net_G, net_F,
                                                                         net_G_ema.ema, net_F_ema.ema,
                                                                         net_G_ema2.ema, net_F_ema2.ema)
            writer.add_scalar('Test/acc', acc1, batch_idx)
            writer.add_scalar('Test/acc_ema', acc2, batch_idx)
            writer.add_scalar('Test/acc_ema2', acc, batch_idx)
            writer.add_scalar('Test/mAcc', per_acc1, batch_idx)
            writer.add_scalar('Test/mAcc_ema', per_acc2, batch_idx)
            writer.add_scalar('Test/mAcc_ema2', per_acc, batch_idx)

            if acc > best_acc_val:
                best_acc_val = acc_val
                cur_acc_test = acc1
                cur_acc_test_ema = acc2
                if args.save:
                    net_F_path = os.path.join(args.log_dir, 'ckpt', 'Best_F.pth')
                    net_F_ema_path = os.path.join(args.log_dir, 'ckpt', 'Best_F_ema.pth')
                    net_F_ema2_path = os.path.join(args.log_dir, 'ckpt', 'Best_F_ema2.pth')
                    net_G_path = os.path.join(args.log_dir, 'ckpt', 'Best_G.pth')
                    net_G_ema_path = os.path.join(args.log_dir, 'ckpt', 'Best_G_ema.pth')
                    net_G_ema2_path = os.path.join(args.log_dir, 'ckpt', 'Best_G_ema2.pth')
                    torch.save(net_G.state_dict(), net_G_path)
                    torch.save(net_F.state_dict(), net_F_path)
                    torch.save(net_G_ema.ema.state_dict(), net_G_ema_path)
                    torch.save(net_F_ema.ema.state_dict(), net_F_ema_path)
                    torch.save(net_G_ema2.ema.state_dict(), net_G_ema2_path)
                    torch.save(net_F_ema2.ema.state_dict(), net_F_ema2_path)
            writer.add_scalar('Test/BestAcc', cur_acc_test, batch_idx)
            writer.add_scalar('Test/BestAcc_ema', cur_acc_test_ema, batch_idx)

            train_meters.reset()


@torch.no_grad()
def test(test_loader, net_G, net_F, iter_idx):
    global args, writer
    net_G.eval()
    net_F.eval()

    correct = 0
    total = 0
    for batch_idx, data_batch in enumerate(tqdm(test_loader)):
        inputs, targets = data_batch[0].cuda(), data_batch[1].cuda()
        _, outputs = net_F(net_G(inputs))

        outputs = torch.softmax(outputs, dim=1)
        max_prob, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    net_G.train()
    net_F.train()
    acc = correct / total * 100.
    return acc


def test_multi(test_loader, net_G1, net_F1, net_G2, net_F2, net_G3, net_F3):
    global args, writer
    net_G1.eval()
    net_F1.eval()
    net_G2.eval()
    net_F2.eval()
    net_G3.eval()
    net_F3.eval()

    correct = 0
    total = 0
    correct1 = 0
    total1 = 0
    correct2 = 0
    total2 = 0

    predicted_list1 = []
    predicted_list2 = []
    predicted_list = []
    target_list = []

    for batch_idx, data_batch in enumerate(tqdm(test_loader)):
        try:
            inputs, targets = data_batch[0].cuda(), data_batch[1].cuda()
        except:
            inputs, targets = data_batch[0][0].cuda(), data_batch[1].cuda()
        _, outputs1 = net_F1(net_G1(inputs))
        _, outputs2 = net_F2(net_G2(inputs))
        _, outputs = net_F3(net_G3(inputs))

        l = 0.5
        outputs1 = torch.softmax(outputs1, dim=1)
        outputs2 = torch.softmax(outputs2, dim=1)
        outputs = torch.softmax(outputs, dim=1)
        # outputs = outputs1 * l + outputs2 * (1 - l)
        max_prob, predicted = outputs.max(1)
        max_prob, predicted1 = outputs1.max(1)
        max_prob, predicted2 = outputs2.max(1)
        total += targets.size(0)

        correct += predicted.eq(targets).sum().item()
        correct1 += predicted1.eq(targets).sum().item()
        correct2 += predicted2.eq(targets).sum().item()

        predicted_list1 += predicted1.cpu().numpy().tolist()
        predicted_list2 += predicted2.cpu().numpy().tolist()
        predicted_list += predicted.cpu().numpy().tolist()
        target_list += targets.cpu().numpy().tolist()

    from sklearn.metrics import confusion_matrix
    cm1 = confusion_matrix(target_list, predicted_list1, normalize='true')
    cm2 = confusion_matrix(target_list, predicted_list2, normalize='true')
    cm = confusion_matrix(target_list, predicted_list, normalize='true')

    per_acc1 = cm1.diagonal().mean() * 100.
    per_acc2 = cm2.diagonal().mean() * 100.
    per_acc = cm.diagonal().mean() * 100.

    net_G1.train()
    net_F1.train()
    net_G2.train()
    net_F2.train()
    acc = correct / total * 100.
    acc1 = correct1 / total * 100.
    acc2 = correct2 / total * 100.
    print('acc1 %.2f acc2 %.2f acc_ens %.2f' % (acc1, acc2, acc))
    print('mAcc1 %.2f mAcc2 %.2f mAcc_ens %.2f' % (per_acc1, per_acc2, per_acc))

    select_cls = np.where(cm1.diagonal() > 0.1)[0]
    if args.num_classes == 12:
        print(cm1.diagonal())
        print(cm2.diagonal())

    return acc1, acc2, acc, per_acc1, per_acc2, per_acc, select_cls


def get_optim_params(model, lr):
    lr_list, lr_multi_list = [], []
    for name, param in model.named_parameters():
        if 'fc' in name:
            lr_multi_list.append(param)
        else:
            lr_list.append(param)
    return [{'params': lr_list, 'lr': lr},
            {'params': lr_multi_list, 'lr': 10 * lr}]


def main():
    global args, writer
    args = get_args()

    print_options(args)

    if args.dataset == 'multi':
        source_loader, target_loader, target_loader_unl, \
            target_loader_val, target_loader_test, class_list = build_dataset(args)
    elif args.dataset == 'office_home':
        source_loader, target_loader, target_loader_unl, \
            target_loader_val, target_loader_test, class_list = build_dataset_officehome(args)
    elif args.dataset == 'visda':
        source_loader, target_loader, target_loader_unl, \
            target_loader_val, target_loader_test, class_list = build_dataset_visda(args)

    # source_loader, _, _, _, _, class_list = build_dataset_tiny(args)

    set_seed(args.seed)
    if args.arch == 'alexnet':
        net_G = AlexNetBase(pret=True).cuda()
        net_F = Predictor(num_class=args.num_classes, temp=args.T).cuda()
        args.inc = 4096

    elif args.arch == 'resnet':
        net_G = resnet34().cuda()

        net_F = Predictor_deep(num_class=args.num_classes, inc=512, temp=args.T)
        # net_F = Predictor_cos_feat(num_class=args.num_classes, inc=512, temp=args.T)
        # net_F = PredictorTSA(num_class=args.num_classes, inc=512)
        weights_init(net_F)
        net_F.cuda()
        args.inc = 512

    optimizer = optim.SGD(get_optim_params(net_G, args.lr) + net_F.get_optim_params(args.lr),
                          momentum=0.9, weight_decay=5e-4, nesterov=True)

    scheduler = InvLr(optimizer)

    tb_dir = os.path.join(args.log_dir, 'tb_log')
    save_dir = os.path.join(args.log_dir, 'ckpt')
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)

    train(source_loader, target_loader, target_loader_unl, target_loader_val, target_loader_test,
          net_G, net_F, optimizer, scheduler)


if __name__ == '__main__':
    main()
