import numpy as np
import torch
import torch.nn.functional as F
import ot


def contras_cls(p1, p2):
    N, C = p1.shape
    cov = p1.t() @ p2

    cov_norm1 = cov / torch.sum(cov, dim=1, keepdims=True)
    cov_norm2 = cov / torch.sum(cov, dim=0, keepdims=True)
    loss = 0.5 * (torch.sum(cov_norm1) - torch.trace(cov_norm1)) / C \
        + 0.5 * (torch.sum(cov_norm2) - torch.trace(cov_norm2)) / C
    return loss


def pairwise_cosine_sim(a, b):
    assert len(a.shape) == 2
    assert a.shape[1] == b.shape[1]
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    mat = a @ b.t()
    return mat


def ot_mapping(M):
    '''
    M: (ns, nt)
    '''
    reg1 = 1
    reg2 = 1
    ns, nt = M.shape
    a, b = np.ones((ns,)) / ns, np.ones((nt,)) / nt
    gamma = ot.unbalanced.sinkhorn_stabilized_unbalanced(a, b, M, reg1, reg2)
    gamma = torch.from_numpy(gamma).cuda()
    return gamma


def ot_loss(proto_s, feat_tu_w, feat_tu_s, args):
    bs = feat_tu_s.shape[0]
    with torch.no_grad():
        M_st_weak = 1 - pairwise_cosine_sim(proto_s.mo_pro, feat_tu_w)  # postive distance
    gamma_st_weak = ot_mapping(M_st_weak.data.cpu().numpy().astype(np.float64), args)
    score_ot, pred_ot = gamma_st_weak.t().max(dim=1)
    Lm = center_loss_cls(proto_s.mo_pro, feat_tu_s, pred_ot, num_classes=args.num_classes)
    return Lm


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


class Prototype:
    def __init__(self, C=65, dim=512, m=0.9):
        self.mo_pro = torch.zeros(C, dim).cuda()
        self.batch_pro = torch.zeros(C, dim).cuda()
        self.m = m

    @torch.no_grad()
    def update(self, feats, lbls, i_iter, norm=False):
        if i_iter < 20:
            momentum = 0
        else:
            momentum = self.m
        # momentum = self.m
        for i_cls in torch.unique(lbls):
            feats_i = feats[lbls == i_cls, :]
            # feats_i = F.normalize(feats_i)
            # feats_i_center = F.normalize(feats_i.mean(dim=0, keepdim=True))
            feats_i_center = feats_i.mean(dim=0, keepdim=True)
            self.mo_pro[i_cls, :] = self.mo_pro[i_cls, :] * \
                momentum + feats_i_center * (1 - momentum)
            self.batch_pro[i_cls, :] = feats_i_center
        if norm:
            self.mo_pro = F.normalize(self.mo_pro)
            self.batch_pro = F.normalize(self.batch_pro)


class MemoryBank:
    def __init__(self, K=8, C=65, dim=512):
        self.mem_feats = torch.randn(C, K, dim).cuda()
        self.K = K
        self.C = torch.zeros(C, K, dtype=torch.long).cuda()
        for i in range(C):
            self.C[i, :] = i

    @torch.no_grad()
    def update(self, feats, lbls):
        for i_cls in torch.unique(lbls):
            feats_i = feats[lbls == i_cls, :]
            feats_i = F.normalize(feats_i)

            bs = feats_i.shape[0]
            if bs >= self.K:
                self.mem_feats[i_cls, :, :] = feats_i[0:self.K, :]
            else:
                self.mem_feats[i_cls, 0:self.K - bs,
                               :] = self.mem_feats[i_cls, bs:, :].clone()
                self.mem_feats[i_cls, self.K - bs:, :] = feats_i


@torch.no_grad()
def ot_mapping_memory(mem_s, feat_t, args):
    K = mem_s.K
    C = mem_s.C
    C = C.reshape(K * args.num_classes)
    mem_feats = mem_s.mem_feats.reshape(K * args.num_classes, -1)
    M = -pairwise_cosine_sim(mem_feats, feat_t) + 1
    ns, nt = M.shape
    a, b = np.ones((ns,)) / ns, np.ones((nt,)) / nt
    gamma = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M.data.cpu().numpy().astype(np.float64), 0.01, 0.5)
    gamma = torch.from_numpy(gamma).cuda().t()  # nt by ns
    return gamma
