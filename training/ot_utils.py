import numpy as np
import torch
import torch.nn.functional as F
import ot


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


def ot_mapping(M):
    '''
    M: (ns, nt)
    '''
    ns, nt = M.shape
    a, b = np.ones((ns,)) / ns, np.ones((nt,)) / nt
    gamma = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M.detach().cpu().numpy().astype(np.float64), 0.01, reg_m=0.5)
    gamma = torch.from_numpy(gamma).cuda()
    return gamma


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


def pairwise_cosine_sim(a, b):
    assert len(a.shape) == 2
    assert a.shape[1] == b.shape[1]
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    mat = a @ b.t()
    return mat
