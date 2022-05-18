import numpy as np
import torch
import torch.nn.functional as F
import ot


def loss_unl(net_G, net_F, imgs_tu_w, imgs_tu_s, proto_s, args):
    '''The proposed losses for unlabeled target samples

    Parameters:
        net_G (network)    --The backbone
        net_F (network)    --The classifier (fc-l2norm-fc)
        imgs_tu_w (tensor) --Weakly augmented inputs
        imgs_tu_s (tensor) --Strongly augmented inputs
        proto_s (tensor)   --Source prototypes

    Return the three losses
    '''
    # forward
    feat_tu = net_G(torch.cat((imgs_tu_w, imgs_tu_s), dim=0))
    feat_tu_con, logits_tu = net_F(feat_tu)
    feat_tu_w, feat_tu_s = feat_tu_con.chunk(2)
    logits_tu_w, logits_tu_s = logits_tu.chunk(2)

    # sample-wise consistency
    pseudo_label = torch.softmax(logits_tu_w.detach() * args.T2, dim=1)
    max_probs, targets_u = torch.max(pseudo_label, dim=1)
    consis_mask = max_probs.ge(args.threshold).float()
    L_pl = (F.cross_entropy(logits_tu_s, targets_u, reduction='none') * consis_mask).mean()

    # class-wise consistency
    prob_tu_w = torch.softmax(logits_tu_w, dim=1)
    prob_tu_s = torch.softmax(logits_tu_s, dim=1)
    L_con_cls = contras_cls(prob_tu_w, prob_tu_s)

    # alignment consistency
    L_ot = ot_loss(proto_s, feat_tu_w, feat_tu_s, args)

    return L_ot, L_con_cls, L_pl, consis_mask


def contras_cls(p1, p2):
    N, C = p1.shape
    cov = p1.t() @ p2

    cov_norm1 = cov / torch.sum(cov, dim=1, keepdims=True)
    cov_norm2 = cov / torch.sum(cov, dim=0, keepdims=True)
    loss = 0.5 * (torch.sum(cov_norm1) - torch.trace(cov_norm1)) / C \
        + 0.5 * (torch.sum(cov_norm2) - torch.trace(cov_norm2)) / C
    return loss


def ot_loss(proto_s, feat_tu_w, feat_tu_s, args):
    bs = feat_tu_s.shape[0]
    with torch.no_grad():
        M_st_weak = 1 - pairwise_cosine_sim(proto_s.mo_pro, feat_tu_w)  # postive distance
    gamma_st_weak = ot_mapping(M_st_weak.data.cpu().numpy().astype(np.float64))
    score_ot, pred_ot = gamma_st_weak.t().max(dim=1)
    Lm = center_loss_cls(proto_s.mo_pro, feat_tu_s, pred_ot, num_classes=args.num_classes)
    return Lm


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
        for i_cls in torch.unique(lbls):
            feats_i = feats[lbls == i_cls, :]
            feats_i_center = feats_i.mean(dim=0, keepdim=True)
            self.mo_pro[i_cls, :] = self.mo_pro[i_cls, :] * \
                momentum + feats_i_center * (1 - momentum)
            self.batch_pro[i_cls, :] = feats_i_center
        if norm:
            self.mo_pro = F.normalize(self.mo_pro)
            self.batch_pro = F.normalize(self.batch_pro)
