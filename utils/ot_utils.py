from numpy.core.fromnumeric import reshape
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import ot
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


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


class Queue:
    def __init__(self, bs=48, num_bs=3, dim=1):
        self.mem_feats = torch.randn(bs * num_bs, dim).cuda()
        self.bs = bs
        self.size = bs * num_bs
        self.ptr = 0

    @torch.no_grad()
    def update(self, feats):
        assert feats.shape[0] == self.bs
        self.mem_feats[self.ptr:self.ptr + self.bs, :] = F.normalize(feats)
        self.ptr = (self.ptr + self.bs) % self.size


class Prototype:
    def __init__(self, C=65, dim=512, m=0.9):
        self.mo_pro = torch.zeros(C, dim).cuda()
        self.batch_pro = torch.zeros(C, dim).cuda()
        self.m = m

    @torch.no_grad()
    def update(self, feats, lbls, i_iter):
        if i_iter == 0:
            momentum = 0
        else:
            momentum = self.m
        for i_cls in torch.unique(lbls):
            feats_i = feats[lbls == i_cls, :]
            feats_i = F.normalize(feats_i)
            feats_i_center = F.normalize(feats_i.mean(dim=0, keepdim=True))
            self.mo_pro[i_cls, :] = self.mo_pro[i_cls, :] * \
                momentum + feats_i_center * (1 - momentum)
            self.batch_pro[i_cls, :] = feats_i_center

        self.mo_pro = F.normalize(self.mo_pro)
        self.batch_pro = F.normalize(self.batch_pro)


def pairwise_cosine_sim(a, b):
    assert len(a.shape) == 2
    assert a.shape[1] == b.shape[1]
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    mat = a @ b.t()
    return mat


def nn_cls(proto, feat, gt):
    M = F.normalize(feat) @ proto.t()
    pred = M.max(dim=1)[1]
    acc = pred.eq(gt).sum().item() / gt.shape[0] * 100.
    return pred, acc


def subcluster_knn(protos, feat, gt):
    M = F.normalize(feat.float()) @ F.normalize(protos.float()).t()
    topk_ind = torch.topk(M, k=8, dim=1)[1].float()
    pred_cls = topk_ind // 8
    major_label, major_index = torch.mode(pred_cls, 1)
    acc = major_label.eq(gt).sum().item() / gt.shape[0] * 100.
    return acc


@torch.no_grad()
def offline_match(source_loader, target_loader, target_loader_unl, net_G, net_F, args):
    net_G.eval()
    net_F.eval()

    source_feats = torch.Tensor().cuda()
    target_feats = torch.Tensor().cuda()
    target_feats_unl = torch.Tensor().cuda()

    source_gt = torch.Tensor().cuda()
    target_gt = torch.Tensor().cuda()
    target_unl_gt = torch.Tensor().cuda()

    source_proto = torch.zeros(args.num_classes, 512).cuda()
    targeproto_t = torch.zeros(args.num_classes, 512).cuda()

    for batch_idx, data_batch in enumerate(tqdm(source_loader)):
        imgs_s, gt_s = data_batch[0].cuda(), data_batch[1].cuda()
        feat_s = net_G(imgs_s)
        source_feats = torch.cat((source_feats, feat_s), dim=0)
        source_gt = torch.cat((source_gt, gt_s), dim=0)
    source_gt = source_gt.cpu().numpy()

    for batch_idx, data_batch in enumerate(tqdm(target_loader)):
        imgs_t, gt_t = data_batch[0].cuda(), data_batch[1].cuda()
        feat_t = net_G(imgs_t)
        target_feats = torch.cat((target_feats, feat_t), dim=0)
        target_gt = torch.cat((target_gt, gt_s), dim=0)

    for batch_idx, data_batch in enumerate(tqdm(target_loader_unl)):
        imgs_t, gt_t = data_batch[0][0].cuda(), data_batch[1].cuda()
        feat_t = net_G(imgs_t)
        target_feats_unl = torch.cat((target_feats_unl, feat_t), dim=0)
        target_unl_gt = torch.cat((target_unl_gt, gt_t), dim=0)

    print('Caculating prototypes...')
    for i_cls in range(args.num_classes):
        source_proto[i_cls, :] = source_feats[source_gt == i_cls, :].mean()
        targeproto_t[i_cls, :] = target_feats[target_gt == i_cls, :].mean()

    # M_s = torch.cdist(source_proto.unsqueeze(0), target_feats_unl.unsqueeze(0), p=2).squeeze()
    # M_s = torch.cdist(source_feats.unsqueeze(0), target_feats_unl.unsqueeze(0), p=2).squeeze()
    # M_t = torch.cdist(targeproto_t.unsqueeze(0), target_feats_unl.unsqueeze(0), p=2).squeeze()
    M_s = 1 - F.normalize(source_feats) @ F.normalize(target_feats_unl).t()

    n_proto_s, n_target = M_s.size(0), M_s.size(1)

    M_s = M_s.cpu().double().numpy()

    print('OT matching')
    a, b = np.ones((n_proto_s,)) / n_proto_s, np.ones((n_target,)) / n_target
    # (ns, nt)
    G_s = ot.sinkhorn(a.astype(np.float64), b.astype(
        np.float64), M_s.astype(np.float64), 10)
    # print(G_s[:, 0].sort())

    gamma_max = np.max(G_s.T, axis=1)
    # gamma_50 = np.percentile(gamma_max, 50)

    # index = gamma_max > gamma_50
    # print(np.sum(index))

    pred_s = np.argmax(G_s.T, axis=1)
    pl_s = source_gt[pred_s]
    pred_s = pl_s
    target_unl_gt = target_unl_gt.cpu().numpy()

    acc = np.sum(pred_s == target_unl_gt) / pred_s.shape[0] * 100.
    # acc = np.sum(pred_s[index] == target_unl_gt[index]) / pred_s[index].shape[0] * 100.
    print('Accuracy is: %.2f' % acc)


@torch.no_grad()
def offline_sample_match(source_loader, target_loader, test_loader, proto_s, net_G, net_F):
    net_G.eval()
    net_F.eval()

    source_feats = torch.Tensor().cuda()
    source_gt = torch.Tensor().cuda()

    target_feats = torch.Tensor().cuda()
    target_gt = torch.Tensor().cuda()

    for batch_idx, data_batch in enumerate(tqdm(source_loader)):
        imgs_s, gt_s = data_batch[0].cuda(), data_batch[1].cuda()
        feat_s, _ = net_F(net_G(imgs_s))
        source_feats = torch.cat((source_feats, feat_s), dim=0)
        source_gt = torch.cat((source_gt, gt_s), dim=0)

    # labeled target samples
    for batch_idx, data_batch in enumerate(tqdm(target_loader)):
        imgs_s, gt_s = data_batch[0].cuda(), data_batch[1].cuda()
        feat_s, _ = net_F(net_G(imgs_s))
        source_feats = torch.cat((source_feats, feat_s), dim=0)
        source_gt = torch.cat((source_gt, gt_s), dim=0)

    for batch_idx, data_batch in enumerate(tqdm(test_loader)):
        imgs_t, gt_t = data_batch[0].cuda(), data_batch[1].cuda()
        # feat_t = net_G(imgs_t)
        feat_t, _ = net_F(net_G(imgs_t))
        target_feats = torch.cat((target_feats, feat_t), dim=0)
        target_gt = torch.cat((target_gt, gt_t), dim=0)

    # prototipical match
    M_s = 1 - pairwise_cosine_sim(target_feats, proto_s)
    ns, nt = M_s.shape
    a, b = np.ones((ns,)) / ns, np.ones((nt,)) / nt
    gamma_s = ot.sinkhorn(a, b, M_s.data.cpu().double().numpy(), 10)
    gamma_s = torch.from_numpy(gamma_s).cuda()
    acc_s = target_gt.eq(gamma_s.max(dim=1)[1]).sum().item() / ns * 100.

    # sample level match
    M = 1 - pairwise_cosine_sim(target_feats, source_feats)
    ns, nt = M.shape
    a, b = np.ones((ns,)) / ns, np.ones((nt,)) / nt
    gamma_s = ot.sinkhorn(a, b, M.data.cpu().double().numpy(), 10)
    gamma_s = torch.from_numpy(gamma_s).cuda()
    pred_sind = gamma_s.max(dim=1)[1]
    pred_lbl = source_gt[pred_sind]
    acc_sample = target_gt.eq(pred_lbl).sum().item() / ns * 100.

    # knn
    knn_acc = nn_cls(proto_s, target_feats, target_gt)[1]

    # source clustering
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import KMeans
    source_feats_np = source_feats.cpu().numpy()
    source_gt_np = source_gt.cpu().numpy()
    N, dim = source_feats_np.shape
    C = 65
    n_subclusters = 4
    source_protos = np.zeros((C, n_subclusters, dim))
    source_protos_gt = np.zeros((C, n_subclusters))
    for i_cls in tqdm(range(C)):
        X = source_feats_np[source_gt_np == i_cls]
        # gmm = GaussianMixture(n_components=n_subclusters).fit(X)
        # source_protos[i_cls, :, :] = gmm.means_
        kmeans = KMeans(n_clusters=n_subclusters).fit(X)
        source_protos[i_cls, :, :] = kmeans.cluster_centers_
        source_protos_gt[i_cls, :] = i_cls
    source_protos = torch.from_numpy(
        source_protos).cuda().reshape(C * n_subclusters, dim)
    source_protos_gt = torch.from_numpy(
        source_protos_gt).cuda().reshape(C * n_subclusters)

    M = 1 - pairwise_cosine_sim(target_feats.float(), source_protos.float())
    ns, nt = M.shape
    a, b = np.ones((ns,)) / ns, np.ones((nt,)) / nt
    gamma_s = ot.sinkhorn(a, b, M.data.cpu().double().numpy(), 10)
    gamma_s = torch.from_numpy(gamma_s).cuda()
    pred_sind = gamma_s.max(dim=1)[1]
    pred_lbl = source_protos_gt[pred_sind]
    acc_subcluster = target_gt.eq(pred_lbl).sum().item() / ns * 100.

    acc_subcluster_knn = subcluster_knn(source_protos, target_feats, target_gt)

    # target cluster
    n_subclusters = 8
    target_feats_np = target_feats.cpu().numpy()
    kmeans = KMeans(n_clusters=n_subclusters * C).fit(target_feats_np)
    target_protos = kmeans.cluster_centers_
    target_protos_labels = torch.from_numpy(kmeans.labels_).cuda().long()
    target_protos = torch.from_numpy(target_protos).cuda()
    target_protos = F.normalize(target_protos)

    M = 1 - pairwise_cosine_sim(target_protos.float(), proto_s)
    ns, nt = M.shape
    a, b = np.ones((ns,)) / ns, np.ones((nt,)) / nt
    G = ot.sinkhorn(a, b, M.data.cpu().double().numpy(), 10)
    G = torch.from_numpy(G).cuda()

    _, pred_ind = G.max(dim=1)  # C * n_sub
    # target_proto_pred = pred_ind  # C * n_sub
    target_pred = pred_ind[target_protos_labels]
    acc_target_cluster = target_gt.eq(
        target_pred).sum().item() / target_pred.shape[0] * 100.

    print('Acc by s proto %.1f sample %.1f knn %.1f knn sub %.1f subcluster %.1f target cluster %.1f' % (
        acc_s, acc_sample, knn_acc, acc_subcluster_knn, acc_subcluster, acc_target_cluster))
    # import pdb
    # pdb.set_trace()


@torch.no_grad()
def offline_multi_level_match(source_loader, target_loader, test_loader, proto_s, net_G, net_F,
                              C=65, level=[2, 4]):

    if len(level) == 0:
        return None, None, None, None
    else:
        level = [int(i) for i in level]

    net_G.eval()
    net_F.eval()

    source_feats = torch.Tensor().cuda()
    source_gt = torch.Tensor().cuda()

    target_feats = torch.Tensor().cuda()
    target_gt = torch.Tensor().cuda()

    pred_list = torch.Tensor().cuda()

    for batch_idx, data_batch in enumerate(tqdm(source_loader)):
        imgs_s, gt_s = data_batch[0].cuda(), data_batch[1].cuda()
        feat_s, _ = net_F(net_G(imgs_s))
        source_feats = torch.cat((source_feats, feat_s), dim=0)
        source_gt = torch.cat((source_gt, gt_s), dim=0)

    # labeled target samples
    for batch_idx, data_batch in enumerate(tqdm(target_loader)):
        imgs_s, gt_s = data_batch[0].cuda(), data_batch[1].cuda()
        feat_s, _ = net_F(net_G(imgs_s))
        source_feats = torch.cat((source_feats, feat_s), dim=0)
        source_gt = torch.cat((source_gt, gt_s), dim=0)

    img_namelist = []
    for batch_idx, data_batch in enumerate(tqdm(test_loader)):
        imgs_t, gt_t = data_batch[0].cuda(), data_batch[1].cuda()
        img_namelist += data_batch[2]
        # feat_t = net_G(imgs_t)
        feat_t, _ = net_F(net_G(imgs_t))
        target_feats = torch.cat((target_feats, feat_t), dim=0)
        target_gt = torch.cat((target_gt, gt_t), dim=0)
    img_namelist = np.array(img_namelist)

    # prototipical match
    M_s = 1 - pairwise_cosine_sim(target_feats, proto_s)
    ns, nt = M_s.shape
    a, b = np.ones((ns,)) / ns, np.ones((nt,)) / nt
    # G_proto = ot.sinkhorn(a, b, M_s.data.cpu().double().numpy(), 10)
    G_proto = ot.unbalanced.sinkhorn_stabilized_unbalanced(a, b, M_s.data.cpu().double().numpy(), 1, 1)
    G_proto = torch.from_numpy(G_proto).cuda()
    pred_proto = G_proto.max(dim=1)[1]
    acc_s = target_gt.eq(pred_proto).sum().item() / ns * 100.

    pred_list = torch.cat((pred_list, pred_proto.unsqueeze(1)), dim=1)

    # source clustering
    source_feats_np = source_feats.cpu().numpy()
    source_gt_np = source_gt.cpu().numpy()
    N, dim = source_feats_np.shape
    C = 65
    n_subclusters = 8
    acc_list = []
    source_proto_list = []
    source_proto_gt_list = []
    for n_subclusters in level:
        source_protos = np.zeros((C, n_subclusters, dim))
        source_protos_gt = np.zeros((C, n_subclusters))
        for i_cls in tqdm(range(C)):
            X = source_feats_np[source_gt_np == i_cls]
            # gmm = GaussianMixture(n_components=n_subclusters).fit(X)
            # source_protos[i_cls, :, :] = gmm.means_
            kmeans = KMeans(n_clusters=n_subclusters).fit(X)
            source_protos[i_cls, :, :] = kmeans.cluster_centers_
            source_protos_gt[i_cls, :] = i_cls
        source_protos = torch.from_numpy(
            source_protos).cuda().reshape(C * n_subclusters, dim)
        source_protos_gt = torch.from_numpy(
            source_protos_gt).cuda().reshape(C * n_subclusters)
        source_proto_list.append(source_protos.reshape(C, n_subclusters, dim))
        source_proto_gt_list.append(source_protos_gt.reshape(C, n_subclusters))

        M = 1 - pairwise_cosine_sim(target_feats.float(), source_protos.float())
        ns, nt = M.shape
        a, b = np.ones((ns,)) / ns, np.ones((nt,)) / nt
        # gamma_s = ot.sinkhorn(a, b, M.data.cpu().double().numpy(), 10)
        gamma_s = ot.unbalanced.sinkhorn_stabilized_unbalanced(a, b, M.data.cpu().double().numpy(), 1, 1)
        gamma_s = torch.from_numpy(gamma_s).cuda()
        pred_sind = gamma_s.max(dim=1)[1]
        pred_subproto = source_protos_gt[pred_sind]
        pred_list = torch.cat((pred_list, pred_subproto.unsqueeze(1)), dim=1)
        acc_subcluster = target_gt.eq(pred_subproto).sum().item() / ns * 100.
        acc_list.append(acc_subcluster)

    print('Acc by s proto %.1f  subcluster %s ' % (acc_s, acc_list))
    net_G.train()
    net_F.train()
    return img_namelist, pred_list, source_proto_list, source_proto_gt_list
