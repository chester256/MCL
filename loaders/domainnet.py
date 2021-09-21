import numpy as np
import os
import os.path
from PIL import Image
from loaders.randaugment import RandAugmentMC
from torchvision import transforms
import pickle
import torch.utils.data as TorchData
from torch.utils.data.distributed import DistributedSampler
from loaders.utils import TransformFixMatch, TransformFixMatchTwo
from loaders.randaugment import RandAugmentMC
from loaders.simsiam_aug import SimSiamTransform, SimSiamTransformTwo
import copy


def get_transforms(crop_size):
    data_transforms = {
        'labeled': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            # RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'fixmatch': TransformFixMatch(crop_size=crop_size),
        'fixmatch_two': TransformFixMatchTwo(crop_size=crop_size),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'simsiam': SimSiamTransform(224),
        'simsiamTwo': SimSiamTransformTwo(224),
    }
    return data_transforms


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def make_dataset_fromlist(image_list):
    with open(image_list) as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]
    with open(image_list) as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            label_list.append(int(label))
            selected_list.append(ind)
        image_index = np.array(image_index)
        label_list = np.array(label_list)
    image_index = image_index[selected_list]
    return image_index, label_list


def return_classlist(image_list):
    with open(image_list) as f:
        label_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[0].split('/')[-2]
            if label not in label_list:
                label_list.append(str(label))
    return label_list


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


class Imagelists_TinyDomainnet(TorchData.Dataset):
    def __init__(self, image_list, data_arr, index_dict, root="./data/multi/",
                 transform=None, target_transform=None, test=False):
        super().__init__()
        imgs, labels = make_dataset_fromlist(image_list)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.test = test
        self.data_arr = data_arr
        self.index_dict = index_dict

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        arr_index = self.index_dict[self.imgs[index]]
        arr = self.data_arr[arr_index]
        img = Image.fromarray(arr)
        target = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)


class Imagelists_Domainnet(TorchData.Dataset):
    def __init__(self, image_list, root="./data/multi/",
                 transform=None, target_transform=None, test=False, repeat=False):
        super().__init__()
        imgs, labels = make_dataset_fromlist(image_list)
        if len(imgs) == 0:
            print(image_list)
        if repeat:
            imgs = imgs.repeat(10)
            labels = labels.repeat(10)
        self.imgs = imgs
        self.labels = labels
        self.is_gt = [1] * len(self.labels)
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.test = test
        self.loader = pil_loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, self.imgs[index], self.is_gt[index]

    def __len__(self):
        return len(self.imgs)


class Imagelists_Domainnet_Repeat(TorchData.Dataset):
    def __init__(self, image_list, root="./data/multi/",
                 transform=None, target_transform=None, test=False, repeat=False):
        super().__init__()
        imgs, labels = make_dataset_fromlist(image_list)
        if len(imgs) == 0:
            print(image_list)
        # if repeat:
        # imgs = imgs.repeat(10)
        # labels = labels.repeat(10)
        self.old_imgs = copy.deepcopy(imgs)
        self.old_labels = copy.deepcopy(labels)
        self.imgs = imgs
        self.labels = labels
        self.is_gt = [1] * len(self.labels)
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.test = test
        self.loader = pil_loader
        self.num_repeat = 50
        self.shuffle_repeat()
        self.is_gt = self.is_gt * self.num_repeat

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, self.imgs[index], self.is_gt[index]

    def __len__(self):
        # return len(self.imgs)
        return len(self.old_imgs) * self.num_repeat

    def shuffle_repeat(self):
        print('Shuffling')
        num_repeat = self.num_repeat
        ind = np.arange(len(self.old_imgs))
        new_imgs = []
        new_labels = []
        for i in range(num_repeat):
            np.random.shuffle(ind)
            new_imgs.append(self.old_imgs[ind])
            new_labels.append(self.old_labels[ind])
        self.imgs = np.concatenate(new_imgs, axis=0)
        self.labels = np.concatenate(new_labels, axis=0)
        print('Finished')


class Pseudo_Domainnet(TorchData.Dataset):
    def __init__(self, imgs, labels, root="./data/multi/",
                 transform=None, target_transform=None, test=False):
        super().__init__()
        # imgs, labels = make_dataset_fromlist(image_list)
        # if len(imgs) == 0:
        #     print(image_list)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.test = test
        self.loader = pil_loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.test:
            return img, target
        else:
            return img, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)


def PL_selection(image_set_file_pseudo_t, sim_cls_path=None, shot=10, scheme='none'):
    imgs, labels = make_dataset_fromlist(image_set_file_pseudo_t)
    new_imgs, new_labels = [], []
    imgs = np.array(imgs)
    labels = np.array(labels)
    sim_cls_list = []
    if len(sim_cls_path) > 0:
        with open(sim_cls_path, 'r') as f:
            for i in f.readlines():
                sim_cls_list.append(int(i.strip()))
    for i in np.unique(labels):
        mask = labels == i
        if len(sim_cls_list) > 0:
            if i not in sim_cls_list:
                shot = int(shot / 2)
            else:
                shot = shot + int(shot / 2)
        new_imgs += imgs[mask][:shot].tolist()
        new_labels += labels[mask][:shot].tolist()

    return new_imgs, new_labels


def build_pseudo_dataset(args):
    base_path = args.base_path
    root = args.data_root
    image_set_file_pseudo_t = args.pseudo_path
    imgs, labels = PL_selection(image_set_file_pseudo_t, sim_cls_path=args.sim_cls_path, shot=args.num_pl, scheme='none')

    image_set_file_t = os.path.join(base_path, 'labeled_target_images_' + args.target + '_%d.txt' % args.num)
    image_set_file_t_val = os.path.join(base_path, 'validation_target_images_' + args.target + '_3.txt')
    image_set_file_unl = os.path.join(base_path, 'unlabeled_target_images_' + args.target + '_%d.txt' % (args.num))

    if args.arch == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = get_transforms(crop_size)

    pseudo_dataset = Pseudo_Domainnet(imgs, labels, root=root, transform=data_transforms[args.labeled_transform])
    # target labeled
    target_dataset = Imagelists_Domainnet(image_set_file_t, root=root, transform=data_transforms[args.labeled_transform])
    # target unlabeled
    target_dataset_unl = Imagelists_Domainnet(image_set_file_unl, root=root, transform=data_transforms[args.unl_transform])
    # target validation
    target_dataset_val = Imagelists_Domainnet(image_set_file_t_val, root=root, transform=data_transforms['val'])
    # target test
    target_dataset_test = Imagelists_Domainnet(image_set_file_unl, root=root, transform=data_transforms['test'])

    class_list = return_classlist(image_set_file_pseudo_t)
    print("%d classes in this dataset" % len(class_list))
    if args.arch == 'alexnet':
        bs = 32
    else:
        bs = 24

    train_sampler = TorchData.RandomSampler if args.local_rank == -1 else DistributedSampler

    drop_last = True
    pseudo_loader = TorchData.DataLoader(pseudo_dataset, sampler=train_sampler(pseudo_dataset),
                                         batch_size=bs, num_workers=args.n_workers, drop_last=drop_last, pin_memory=True)
    target_loader = TorchData.DataLoader(target_dataset, sampler=train_sampler(target_dataset),
                                         batch_size=min(bs, len(target_dataset)),
                                         num_workers=args.n_workers, drop_last=drop_last, pin_memory=True)
    target_loader_unl = TorchData.DataLoader(target_dataset_unl, sampler=train_sampler(target_dataset_unl),
                                             batch_size=bs * args.bs_unl_multi, num_workers=args.n_workers, drop_last=drop_last, pin_memory=True)

    target_loader_val = TorchData.DataLoader(target_dataset_val, sampler=TorchData.SequentialSampler(target_dataset_val), batch_size=bs * 2,
                                             num_workers=args.n_workers, drop_last=False, pin_memory=True)

    target_loader_test = TorchData.DataLoader(target_dataset_test, sampler=TorchData.SequentialSampler(target_dataset_test), batch_size=bs * 2,
                                              num_workers=args.n_workers, drop_last=False, pin_memory=True)

    return pseudo_loader, target_loader, target_loader_unl, \
        target_loader_val, target_loader_test, class_list


def build_dataset(args):
    base_path = args.base_path
    root = args.data_root
    image_set_file_s = os.path.join(base_path, 'labeled_source_images_' + args.source + '.txt')

    image_set_file_t = os.path.join(base_path, 'labeled_target_images_' + args.target + '_%d.txt' % args.num)
    image_set_file_t_val = os.path.join(base_path, 'validation_target_images_' + args.target + '_3.txt')
    image_set_file_unl = os.path.join(base_path, 'unlabeled_target_images_' + args.target + '_%d.txt' % (args.num))

    if args.arch == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224

    data_transforms = get_transforms(crop_size)

    source_dataset = Imagelists_Domainnet(image_set_file_s, root=root, transform=data_transforms[args.labeled_transform])
    # target labeled
    # target_dataset = Imagelists_Domainnet(image_set_file_t, root=root, transform=data_transforms[args.labeled_transform], repeat=True)
    target_dataset = Imagelists_Domainnet_Repeat(image_set_file_t, root=root, transform=data_transforms[args.labeled_transform])
    target_dataset.shuffle_repeat()
    # target unlabeled
    target_dataset_unl = Imagelists_Domainnet(image_set_file_unl, root=root, transform=data_transforms[args.unl_transform])
    # target validation
    target_dataset_val = Imagelists_Domainnet(image_set_file_t_val, root=root, transform=data_transforms['val'])
    # target test
    target_dataset_test = Imagelists_Domainnet(image_set_file_unl, root=root, transform=data_transforms['test'])

    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    if args.arch == 'alexnet':
        bs = 32
    else:
        bs = 24

    train_sampler = TorchData.RandomSampler if args.local_rank == -1 else DistributedSampler

    drop_last = True
    bs = args.bs
    source_loader = TorchData.DataLoader(source_dataset, sampler=train_sampler(source_dataset),
                                         batch_size=bs, num_workers=args.n_workers, drop_last=drop_last, pin_memory=True)
    # target_loader = TorchData.DataLoader(target_dataset, sampler=train_sampler(target_dataset),
    #                                      batch_size=min(bs, len(target_dataset)),
    #                                      num_workers=args.n_workers, drop_last=drop_last, pin_memory=True)
    target_loader = TorchData.DataLoader(target_dataset, batch_size=min(bs, len(target_dataset)), shuffle=False,
                                         num_workers=args.n_workers, drop_last=drop_last, pin_memory=True)
    target_loader_unl = TorchData.DataLoader(target_dataset_unl, sampler=train_sampler(target_dataset_unl),
                                             batch_size=bs * args.bs_unl_multi, num_workers=args.n_workers, drop_last=drop_last, pin_memory=True)

    target_loader_val = TorchData.DataLoader(target_dataset_val, sampler=TorchData.SequentialSampler(target_dataset_val), batch_size=bs * 2,
                                             num_workers=args.n_workers, drop_last=False, pin_memory=True)

    target_loader_test = TorchData.DataLoader(target_dataset_test, sampler=TorchData.SequentialSampler(target_dataset_test), batch_size=bs * 2,
                                              num_workers=args.n_workers, drop_last=False, pin_memory=True)

    return source_loader, target_loader, target_loader_unl, \
        target_loader_val, target_loader_test, class_list


def build_dataset_tiny(args):
    # base_path = args.base_path
    root = args.data_root
    base_path = './data/txt/TinyDomainnet/'
    image_set_file_s = os.path.join(base_path, 'labeled_source_images_' + args.source + '.txt')

    image_set_file_t = os.path.join(base_path, 'labeled_target_images_' + args.target + '_%d.txt' % args.num)
    image_set_file_t_val = os.path.join(base_path, 'validation_target_images_' + args.target + '_3.txt')
    image_set_file_unl = os.path.join(base_path, 'unlabeled_target_images_' + args.target + '_%d.txt' % (args.num))

    if args.arch == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = get_transforms(crop_size)

    # load np array
    print('Loading all data...')
    data_arr_s = np.load('/data/zizheng/domainnet/TinyDomainnet/%s.npy' % args.source)
    data_arr_t = np.load('/data/zizheng/domainnet/TinyDomainnet/%s.npy' % args.target)
    with open('/data/zizheng/domainnet/TinyDomainnet/%s.pkl' % args.source, 'rb') as f:
        index_dict_s = pickle.load(f)
    with open('/data/zizheng/domainnet/TinyDomainnet/%s.pkl' % args.target, 'rb') as f:
        index_dict_t = pickle.load(f)
    print('Finish load')

    source_dataset = Imagelists_TinyDomainnet(image_set_file_s, data_arr_s, index_dict_s, root=root, transform=data_transforms[args.labeled_transform])
    # target labeled
    target_dataset = Imagelists_TinyDomainnet(image_set_file_t, data_arr_t, index_dict_t, root=root, transform=data_transforms[args.labeled_transform])
    # target unlabeled
    # target_dataset_unl = Imagelists_TinyDomainnet(image_set_file_unl, data_arr_t, index_dict_t, root=root, transform=data_transforms['fixmatch'])
    target_dataset_unl = Imagelists_TinyDomainnet(image_set_file_unl, data_arr_t, index_dict_t, root=root, transform=data_transforms[args.unl_transform])
    # target validation
    target_dataset_val = Imagelists_TinyDomainnet(image_set_file_t_val, data_arr_t, index_dict_t, root=root, transform=data_transforms['val'])
    # target test
    target_dataset_test = Imagelists_TinyDomainnet(image_set_file_unl, data_arr_t, index_dict_t, root=root, transform=data_transforms['test'])

    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    if args.arch == 'alexnet':
        bs = 32
    else:
        bs = args.bs

    train_sampler = TorchData.RandomSampler if args.local_rank == -1 else DistributedSampler

    drop_last = True
    n_workers = args.n_workers
    source_loader = TorchData.DataLoader(source_dataset, sampler=train_sampler(source_dataset),
                                         batch_size=bs, num_workers=n_workers, drop_last=drop_last, pin_memory=True)
    target_loader = TorchData.DataLoader(target_dataset, sampler=train_sampler(target_dataset),
                                         batch_size=min(bs, len(target_dataset)),
                                         num_workers=n_workers, drop_last=drop_last, pin_memory=True)
    target_loader_unl = TorchData.DataLoader(target_dataset_unl, sampler=train_sampler(target_dataset_unl),
                                             batch_size=bs * 2, num_workers=n_workers, drop_last=drop_last, pin_memory=True)

    target_loader_val = TorchData.DataLoader(target_dataset_val, sampler=TorchData.SequentialSampler(target_dataset_val), batch_size=bs * 2,
                                             num_workers=n_workers, drop_last=False, pin_memory=True)

    target_loader_test = TorchData.DataLoader(target_dataset_test, sampler=TorchData.SequentialSampler(target_dataset_test), batch_size=bs * 2,
                                              num_workers=n_workers, drop_last=False, pin_memory=True)

    return source_loader, target_loader, target_loader_unl, \
        target_loader_val, target_loader_test, class_list


if __name__ == '__main__':
    import argparse
    import time
    parser = argparse.ArgumentParser(description='Domain label estimation')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--source', type=str, default='real')
    parser.add_argument('--target', type=str, default='sketch')
    parser.add_argument('--bs', default=24, type=int)
    parser.add_argument('--n_workers', default=6, type=int)
    parser.add_argument('--data_root', type=str, default='/data/zizheng/domainnet')
    parser.add_argument('--unl_transform', type=str, default='fixmatch')
    parser.add_argument('--num', type=int, default=3,
                        help='number of labeled examples in the target')
    parser.add_argument('--dataset', type=str, default='office_home',
                        choices=['multi', 'office', 'office_home'],
                        help='the name of dataset')
    parser.add_argument('--arch', type=str, default='resnet',
                        help='which network to use')
    parser.add_argument('--base_path', type=str, default='./data/txt/multi/')
    args = parser.parse_args()

    source_loader, target_loader, target_loader_unl, \
        target_loader_val, target_loader_test, class_list = build_dataset(args)

    start = time.time()
    for batch_idx in range(100):
        try:
            data_batch_source = source_iter.next()
            data_batch_target = target_iter.next()
            data_batch_unl = next(target_unl_iter)
        except:
            source_iter = iter(source_loader)
            target_iter = iter(target_loader)
            target_unl_iter = iter(target_loader_unl)

            data_batch_source = source_iter.next()
            data_batch_target = target_iter.next()
            data_batch_unl = next(target_unl_iter)

        imgs_s, gt_s = data_batch_source[0].cuda(), data_batch_source[1].cuda()
        imgs_t, gt_t = data_batch_target[0].cuda(), data_batch_target[1].cuda()

        imgs_tu_w, imgs_tu_s = data_batch_unl[0][0].cuda(), data_batch_unl[0][1].cuda()
        gt_tu = data_batch_unl[1].cuda()

        time_e = time.time() - start
        print('Batch time: ', time_e)
        start = time.time()
