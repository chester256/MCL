import os
import torch
import torch.nn as nn
import shutil
import os.path as osp
import numpy as np
import random
from glob import glob
import time

from torch.nn.modules import loss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.1)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.fill_(0)


def save_checkpoint(state, is_best, checkpoint='checkpoint',
                    filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AllMeters:
    def __init__(self, name_list) -> None:
        self.meter_dict = {}
        self.name_list = name_list
        for name in name_list:
            self.meter_dict[name] = AverageMeter()
        self.last_time = time.time()

    def add(self, name_list):
        for name in name_list:
            if name in self.meter_dict:
                print('Already in!')
                raise NameError
            self.meter_dict[name] = AverageMeter()
            self.name_list.append(name)

    def update(self, k, v, n=1):
        self.meter_dict[k].update(v, n)

    def update_list(self, k_list, v_list):
        for k, v in zip(k_list, v_list):
            self.meter_dict[k].update(v)

    def get(self, k):
        return self.meter_dict[k].avg

    def reset(self):
        self.meter_dict = {}
        for name in self.name_list:
            self.meter_dict[name] = AverageMeter()

    def tb_log(self, writer, step, prefix='Train'):
        for name in self.name_list:
            tb_name = prefix + '/' + name
            v = self.get(name)
            writer.add_scalar(tb_name, v, step)

    def log_str(self, step, args):
        s = args.source.upper()[0]
        t = args.target.upper()[0]
        cur_time = time.time()
        log = '%s -> %s Iter %d Time %.2f ' % (s, t, step, cur_time - self.last_time)
        for k in self.name_list:
            v = self.get(k)
            log += '%s %.4f ' % (k, v)
        self.last_time = time.time()
        return log


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def print_options(args):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    os.makedirs(args.log_dir, exist_ok=True)
    file_name = osp.join(args.log_dir, 'opt.txt')
    with open(file_name, 'wt') as args_file:
        args_file.write(message)
        args_file.write('\n')


def officehome2pkl():
    from PIL import Image
    import pickle
    from tqdm import tqdm
    root = '/home/yanzizheng/SSDA_MME/data/office_home'
    pattern = root + '/**/*.jpg'
    img_list = glob(pattern, recursive=True)
    name_list = [i[len(root) + 1:] for i in img_list]
    index_dict = {}
    img_save_list = []
    for index, path in enumerate(tqdm(img_list)):
        img = Image.open(path).convert('RGB')
        print(img.size)
        # img = Image.open(path).convert('RGB').resize((256, 256))
        # img = np.array(img, dtype=np.uint8)
        # img_save_list.append(img)
        # index_dict[name_list[index]] = index

    # print('Saving pkl')

    # img_save_list = np.array(img_save_list)
    # np.save('/data/zizheng/office_home_image.npy', img_save_list)

    # with open('/data/zizheng/office_home_index.pkl', 'wb') as f:
    #     pickle.dump(index_dict, f)


def domainnet2pkl(domain):
    from PIL import Image
    import pickle
    from tqdm import tqdm
    root = '/data/zizheng/domainnet/%s' % domain
    pattern = root + '/**/*.jpg'
    img_list = glob(pattern, recursive=True)
    name_list = [i[len(root) + 1:] for i in img_list]
    index_dict = {}
    img_save_list = []
    for index, path in enumerate(tqdm(img_list)):
        img = Image.open(path).convert('RGB').resize((256, 256))
        # img = Image.open(path).convert('RGB')
        # print(img.size)
        img = np.array(img, dtype=np.uint8)
    #     img_save_list.append(img)
    #     index_dict[name_list[index]] = index

    # print('Saving pkl')
    # img_save_list = np.array(img_save_list)
    # np.save('/data/zizheng/domainnet/domainnet.npy', img_save_list)

    # with open('/data/zizheng/domainnet/domainnet.pkl', 'wb') as f:
    #     pickle.dump(index_dict, f)


def cal_noise_matrix():
    from sklearn.metrics import confusion_matrix
    pl_file_path = '/home/yanzizheng/SSDA_MME/logs/dividemix_A/pl/unlabeled_target_images_Art_1.txt'
    gt_file_path = '/home/yanzizheng/SSDA_MME/data/txt/office_home/unlabeled_target_images_Art_1.txt'
    with open(gt_file_path) as f:
        label_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            label_list.append(int(label))

    with open(pl_file_path) as f:
        pl_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            pl_list.append(int(label))

    print(confusion_matrix(label_list, pl_list))


class DataMemory:
    def __init__(self, K=2, C=65):
        self.mem = torch.randn(C, K, 3, 224, 224)
        self.C = C
        self.K = K

    def update(self, imgs, gts):
        for i_cls in torch.unique(gts):
            img_i = imgs[gts == i_cls]
            bs = img_i.shape[0]

            if bs >= self.K:
                self.mem[i_cls, :, :, :, :] = img_i[0:self.K, :, :, :]
            else:
                self.mem[i_cls, 0:self.K - bs, :, :, :] = self.mem[i_cls, bs:, :, :, :].clone()
                self.mem[i_cls, self.K - bs:, :, :, :] = img_i


def cls2name():
    import json
    file_path = r'E:\zzyan\research\SSDA\MTDA\data\txt\multi\labeled_target_images_clipart_1.txt'
    with open(file_path, 'r') as f:
        lines = f.readlines()
    cls2name_dict = {}
    for i in lines:
        line = i.split()
        name = line[0].split('/')[1]
        cls_id = line[1]
        print(name, cls_id)
        cls2name_dict[cls_id] = name

    with open('cls2name.json', 'w') as f:
        json.dump(cls2name_dict, f)


if __name__ == '__main__':
    # officehome2pkl()
    # domain = 'sketch'
    # domainnet2pkl(domain)
    cls2name()
