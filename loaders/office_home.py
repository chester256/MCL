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
from loaders.simsiam_aug import SimSiamTransform
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
        # 'fixmatch': TransformFixMatchTwo(crop_size=crop_size),
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


class Imagelists_OfficeHome(TorchData.Dataset):
    def __init__(self, image_list, data_arr, index_dict, root="./data/multi/",
                 transform=None, target_transform=None, test=False, domain_ind=None, repeat=False):
        super().__init__()
        imgs, labels = make_dataset_fromlist(image_list)
        if repeat:
            imgs = imgs.repeat(10)
            labels = labels.repeat(10)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.test = test
        self.data_arr = data_arr
        self.index_dict = index_dict
        self.domain_ind = domain_ind

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

        if self.domain_ind is not None:
            return img, target, self.imgs[index], self.domain_ind
        return img, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)


class Imagelists_OfficeHome_Repeat(TorchData.Dataset):
    def __init__(self, image_list, data_arr, index_dict, root="./data/multi/",
                 transform=None, target_transform=None, test=False, domain_ind=None):
        super().__init__()
        imgs, labels = make_dataset_fromlist(image_list)
        self.old_imgs = copy.deepcopy(imgs)
        self.old_labels = copy.deepcopy(labels)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.test = test
        self.data_arr = data_arr
        self.index_dict = index_dict
        self.domain_ind = domain_ind
        self.num_repeat = 500
        self.shuffle_repeat()

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

        if self.domain_ind is not None:
            return img, target, self.imgs[index], self.domain_ind
        return img, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)

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


class Imagelists_OfficeHome_SimSiam(TorchData.Dataset):
    def __init__(self, image_list, data_arr, index_dict, root="./data/multi/",
                 transform=None, target_transform=None, test=False, domain_ind=None):
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
        self.domain_ind = domain_ind
        self.simsiam_transform = SimSiamTransform(image_size=224)

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
            img0 = self.transform(img)

        img1, img2 = self.simsiam_transform(img)
        img = (img0, img1, img2)

        if self.domain_ind is not None:
            return img, target, self.imgs[index], self.domain_ind
        return img, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)


class Imagelists_OfficeHome_Divide(TorchData.Dataset):
    def __init__(self, image_list, data_arr, index_dict, root="./data/multi/",
                 transform=None, target_transform=None, test=False, domain_ind=None,
                 pred=None, probability=None, mode=None):
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
        self.domain_ind = domain_ind
        self.mode = mode

        if self.mode == 'labeled':
            pred_idx = pred.nonzero()[0]
            self.probability = [probability[i] for i in pred_idx]
            self.imgs = [self.imgs[i] for i in pred_idx]
            self.labels = [self.labels[i] for i in pred_idx]
        elif self.mode == "unlabeled":
            pred_idx = (1 - pred).nonzero()[0]
            self.imgs = [self.imgs[i] for i in pred_idx]
            self.labels = [self.labels[i] for i in pred_idx]
        elif self.mode == 'gt_labeled':
            self.probability = [1.] * len(self.imgs)

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

        if self.mode == 'unlabeled':
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, target
        elif self.mode == 'labeled' or self.mode == 'gt_labeled':
            img1 = self.transform(img)
            img2 = self.transform(img)
            prob = self.probability[index]
            return img1, img2, target, prob
        elif self.model == 'test':
            img = self.transform(img)
            return img, target

    def __len__(self):
        return len(self.imgs)


def build_dataset_officehome(args):
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

    # load np array
    data_arr = np.load('/data/zizheng/office_home_image.npy')
    with open('/data/zizheng/office_home_index.pkl', 'rb') as f:
        index_dict = pickle.load(f)

    data_transforms = get_transforms(crop_size)

    source_dataset = Imagelists_OfficeHome(image_set_file_s, data_arr, index_dict, root=root, transform=data_transforms['labeled'])

    # target labeled
    # target_dataset = Imagelists_OfficeHome(image_set_file_t, data_arr, index_dict, root=root,
    #                                        transform=data_transforms[args.labeled_transform], repeat=True)
    target_dataset = Imagelists_OfficeHome_Repeat(image_set_file_t, data_arr, index_dict, root=root,
                                                  transform=data_transforms[args.labeled_transform])
    target_dataset.shuffle_repeat()
    # target unlabeled
    target_dataset_unl = Imagelists_OfficeHome(image_set_file_unl, data_arr, index_dict, root=root, transform=data_transforms[args.unl_transform])
    # target validation
    target_dataset_val = Imagelists_OfficeHome(image_set_file_t_val, data_arr, index_dict, root=root, transform=data_transforms['val'])
    # target test
    target_dataset_test = Imagelists_OfficeHome(image_set_file_unl, data_arr, index_dict, root=root, transform=data_transforms['test'])

    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))

    train_sampler = TorchData.RandomSampler if args.local_rank == -1 else DistributedSampler

    drop_last = True
    bs = args.bs
    source_loader = TorchData.DataLoader(source_dataset, sampler=train_sampler(source_dataset),
                                         batch_size=bs, num_workers=args.n_workers, drop_last=drop_last, pin_memory=True)
    target_loader = TorchData.DataLoader(target_dataset, shuffle=False,
                                         batch_size=min(bs, len(target_dataset)),
                                         num_workers=args.n_workers, drop_last=drop_last, pin_memory=True)
    target_loader_unl = TorchData.DataLoader(target_dataset_unl, sampler=train_sampler(target_dataset_unl),
                                             batch_size=bs * args.bs_unl_multi, num_workers=args.n_workers, drop_last=drop_last, pin_memory=True)

    target_loader_val = TorchData.DataLoader(target_dataset_val, sampler=TorchData.SequentialSampler(target_dataset_val), batch_size=bs * 2,
                                             num_workers=args.n_workers, drop_last=False, pin_memory=True)

    target_loader_test = TorchData.DataLoader(target_dataset_test, sampler=TorchData.SequentialSampler(target_dataset_test), batch_size=bs * 2,
                                              num_workers=args.n_workers, drop_last=False, pin_memory=True)

    return source_loader, target_loader, target_loader_unl, \
        target_loader_val, target_loader_test, class_list


def build_dataset(args):
    base_path = './data/txt/%s' % args.dataset
    root = args.data_root
    image_set_file_s = os.path.join(base_path, 'labeled_source_images_' + args.source + '.txt')

    image_set_file_t_list = [os.path.join(base_path, 'labeled_target_images_' + target + '_%d.txt' % (args.num)) for target in args.target_list]
    image_set_file_t_val_list = [os.path.join(base_path, 'validation_target_images_' + target + '_3.txt') for target in args.target_list]
    image_set_file_unl_list = [os.path.join(base_path, 'unlabeled_target_images_' + target + '_%d.txt' % (args.num)) for target in args.target_list]

    if args.domain_ind:
        domain_ind_list = [i for i in range(len(args.target_list))]
    else:
        domain_ind_list = [None] * len(args.target_list)

    if args.arch == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224

    data_transforms = get_transforms(crop_size)

    # load np array
    data_arr = np.load('/data/zizheng/office_home_image.npy')
    with open('/data/zizheng/office_home_index.pkl', 'rb') as f:
        index_dict = pickle.load(f)

    source_dataset = Imagelists_OfficeHome(image_set_file_s, data_arr, index_dict, root=root, transform=data_transforms['labeled'])

    # target labeled
    target_dataset_list = []
    for i, domain_ind in zip(image_set_file_t_list, domain_ind_list):
        dset = Imagelists_OfficeHome(i, data_arr, index_dict, root=root,
                                     transform=data_transforms['labeled'], domain_ind=domain_ind)
        target_dataset_list.append(dset)

    # target unlabeled
    target_dataset_unl_list = []
    for i, domain_ind in zip(image_set_file_unl_list, domain_ind_list):
        dset = Imagelists_OfficeHome(i, data_arr, index_dict, root=root,
                                     transform=data_transforms['fixmatch'], domain_ind=domain_ind)
        target_dataset_unl_list.append(dset)

    # target validation
    target_dataset_val_list = []
    for i in image_set_file_t_val_list:
        dset = Imagelists_OfficeHome(i, data_arr, index_dict, root=root, transform=data_transforms['val'])
        target_dataset_val_list.append(dset)

    # target test
    target_dataset_test_list = []
    for i in image_set_file_unl_list:
        dset = Imagelists_OfficeHome(i, data_arr, index_dict, root=root, transform=data_transforms['test'])
        target_dataset_test_list.append(dset)

    target_dataset = TorchData.ConcatDataset(target_dataset_list)
    target_dataset_unl = TorchData.ConcatDataset(target_dataset_unl_list)

    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    if args.arch == 'alexnet':
        bs = 32
    else:
        bs = 24

    train_sampler = TorchData.RandomSampler if args.local_rank == -1 else DistributedSampler

    drop_last = True
    source_loader = TorchData.DataLoader(source_dataset, sampler=train_sampler(source_dataset),
                                         batch_size=bs, num_workers=4, drop_last=drop_last, pin_memory=True)
    target_loader = TorchData.DataLoader(target_dataset, sampler=train_sampler(target_dataset),
                                         batch_size=min(bs, len(target_dataset)),
                                         num_workers=4, drop_last=drop_last, pin_memory=True)
    target_loader_unl = TorchData.DataLoader(target_dataset_unl, sampler=train_sampler(target_dataset_unl),
                                             batch_size=bs * 2, num_workers=4, drop_last=drop_last, pin_memory=True)

    target_loader_val_list = []
    for dataset in target_dataset_val_list:
        loader = TorchData.DataLoader(dataset, sampler=TorchData.SequentialSampler(dataset), batch_size=bs * 2,
                                      num_workers=4, drop_last=False, pin_memory=True)
        target_loader_val_list.append(loader)

    target_loader_test_list = []
    for dataset in target_dataset_test_list:
        loader = TorchData.DataLoader(dataset, sampler=TorchData.SequentialSampler(dataset), batch_size=bs * 2,
                                      num_workers=4, drop_last=False, pin_memory=True)
        target_loader_test_list.append(loader)

    return source_loader, target_loader, target_loader_unl, \
        target_loader_val_list, target_loader_test_list, class_list


def build_dataset_simsiam(args):
    base_path = './data/txt/%s' % args.dataset
    root = args.data_root
    image_set_file_s = os.path.join(base_path, 'labeled_source_images_' + args.source + '.txt')

    image_set_file_t_list = [os.path.join(base_path, 'labeled_target_images_' + target + '_%d.txt' % (args.num)) for target in args.target_list]
    image_set_file_t_val_list = [os.path.join(base_path, 'validation_target_images_' + target + '_3.txt') for target in args.target_list]
    image_set_file_unl_list = [os.path.join(base_path, 'unlabeled_target_images_' + target + '_%d.txt' % (args.num)) for target in args.target_list]

    if args.domain_ind:
        domain_ind_list = [i for i in range(len(args.target_list))]
    else:
        domain_ind_list = [None] * len(args.target_list)

    if args.arch == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224

    data_transforms = get_transforms(crop_size)

    # load np array
    data_arr = np.load('/data/zizheng/office_home_image.npy')
    with open('/data/zizheng/office_home_index.pkl', 'rb') as f:
        index_dict = pickle.load(f)

    source_dataset = Imagelists_OfficeHome_SimSiam(image_set_file_s, data_arr, index_dict, root=root, transform=data_transforms['labeled'])

    # target labeled
    target_dataset_list = []
    for i, domain_ind in zip(image_set_file_t_list, domain_ind_list):
        dset = Imagelists_OfficeHome_SimSiam(i, data_arr, index_dict, root=root,
                                             transform=data_transforms['labeled'], domain_ind=domain_ind)
        target_dataset_list.append(dset)

    # target unlabeled
    target_dataset_unl_list = []
    for i, domain_ind in zip(image_set_file_unl_list, domain_ind_list):
        dset = Imagelists_OfficeHome_SimSiam(i, data_arr, index_dict, root=root,
                                             transform=data_transforms['fixmatch'], domain_ind=domain_ind)
        target_dataset_unl_list.append(dset)

    # target validation
    target_dataset_val_list = []
    for i in image_set_file_t_val_list:
        dset = Imagelists_OfficeHome_SimSiam(i, data_arr, index_dict, root=root, transform=data_transforms['val'])
        target_dataset_val_list.append(dset)

    # target test
    target_dataset_test_list = []
    for i in image_set_file_unl_list:
        dset = Imagelists_OfficeHome_SimSiam(i, data_arr, index_dict, root=root, transform=data_transforms['test'])
        target_dataset_test_list.append(dset)

    target_dataset = TorchData.ConcatDataset(target_dataset_list)
    target_dataset_unl = TorchData.ConcatDataset(target_dataset_unl_list)

    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    if args.arch == 'alexnet':
        bs = 32
    else:
        bs = 24

    train_sampler = TorchData.RandomSampler if args.local_rank == -1 else DistributedSampler

    source_loader = TorchData.DataLoader(source_dataset, sampler=train_sampler(source_dataset),
                                         batch_size=bs, num_workers=4, drop_last=True, pin_memory=True)
    target_loader = TorchData.DataLoader(target_dataset, sampler=train_sampler(target_dataset),
                                         batch_size=min(bs, len(target_dataset)),
                                         num_workers=4, drop_last=True, pin_memory=True)
    target_loader_unl = TorchData.DataLoader(target_dataset_unl, sampler=train_sampler(target_dataset_unl),
                                             batch_size=bs * 2, num_workers=4, drop_last=True, pin_memory=True)

    target_loader_val_list = []
    for dataset in target_dataset_val_list:
        loader = TorchData.DataLoader(dataset, sampler=TorchData.SequentialSampler(dataset), batch_size=bs * 2,
                                      num_workers=4, drop_last=False, pin_memory=True)
        target_loader_val_list.append(loader)

    target_loader_test_list = []
    for dataset in target_dataset_test_list:
        loader = TorchData.DataLoader(dataset, sampler=TorchData.SequentialSampler(dataset), batch_size=bs * 2,
                                      num_workers=4, drop_last=False, pin_memory=True)
        target_loader_test_list.append(loader)

    return source_loader, target_loader, target_loader_unl, \
        target_loader_val_list, target_loader_test_list, class_list


def build_dataset_cotraining(args):
    base_path = './data/txt/%s' % args.dataset
    root = args.data_root
    image_set_file_s = os.path.join(base_path, 'labeled_source_images_' + args.source + '.txt')

    image_set_file_t_list = [os.path.join(base_path, 'labeled_target_images_' + target + '_%d.txt' % (args.num)) for target in args.target_list]
    image_set_file_t_val_list = [os.path.join(base_path, 'validation_target_images_' + target + '_3.txt') for target in args.target_list]
    image_set_file_unl_list = [os.path.join(args.dom_pred_path, 'unlabeled_target_images_' + target + '_%d.txt' % (args.num)) for target in args.target_list]
    image_set_file_unl_test_list = [os.path.join(base_path, 'unlabeled_target_images_' + target + '_%d.txt' % (args.num)) for target in args.target_list]

    if args.domain_ind:
        domain_ind_list = [i for i in range(len(args.target_list))]
    else:
        domain_ind_list = [None] * len(args.target_list)

    if args.arch == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224

    data_transforms = get_transforms(crop_size)

    # load np array
    data_arr = np.load('/data/zizheng/office_home_image.npy')
    with open('/data/zizheng/office_home_index.pkl', 'rb') as f:
        index_dict = pickle.load(f)

    source_dataset = Imagelists_OfficeHome(image_set_file_s, data_arr, index_dict, root=root, transform=data_transforms['labeled'])

    # target labeled
    target_dataset_list = []
    for i, domain_ind in zip(image_set_file_t_list, domain_ind_list):
        dset = Imagelists_OfficeHome(i, data_arr, index_dict, root=root,
                                     transform=data_transforms['labeled'], domain_ind=domain_ind)
        target_dataset_list.append(dset)

    # target unlabeled
    target_dataset_unl_list = []
    for i, domain_ind in zip(image_set_file_unl_list, domain_ind_list):
        dset = Imagelists_OfficeHome(i, data_arr, index_dict, root=root,
                                     transform=data_transforms['fixmatch'], domain_ind=domain_ind)
        target_dataset_unl_list.append(dset)

    # target validation
    target_dataset_val_list = []
    for i in image_set_file_t_val_list:
        dset = Imagelists_OfficeHome(i, data_arr, index_dict, root=root, transform=data_transforms['val'])
        target_dataset_val_list.append(dset)

    # target test
    target_dataset_test_list = []
    for i in image_set_file_unl_test_list:
        dset = Imagelists_OfficeHome(i, data_arr, index_dict, root=root, transform=data_transforms['test'])
        target_dataset_test_list.append(dset)

    target_dataset = TorchData.ConcatDataset(target_dataset_list)
    # target_dataset_unl = TorchData.ConcatDataset(target_dataset_unl_list)

    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    if args.arch == 'alexnet':
        bs = 32
    else:
        bs = 24

    train_sampler = TorchData.RandomSampler if args.local_rank == -1 else DistributedSampler

    source_loader = TorchData.DataLoader(source_dataset, shuffle=True,
                                         batch_size=bs, num_workers=4, drop_last=True, pin_memory=True)
    target_loader = TorchData.DataLoader(target_dataset, shuffle=True,
                                         batch_size=min(bs, len(target_dataset)),
                                         num_workers=4, drop_last=True, pin_memory=True)

    source_loader_copy = TorchData.DataLoader(source_dataset, shuffle=True,
                                              batch_size=bs, num_workers=4, drop_last=True, pin_memory=True)
    target_loader_copy = TorchData.DataLoader(target_dataset, shuffle=True,
                                              batch_size=min(bs, len(target_dataset)),
                                              num_workers=4, drop_last=True, pin_memory=True)

    # target_loader_unl = TorchData.DataLoader(target_dataset_unl, sampler=train_sampler(target_dataset_unl),
    #                                          batch_size=bs * 2, num_workers=4, drop_last=True, pin_memory=True)
    target_loader_unl_list = []
    for dataset in target_dataset_unl_list:
        loader = TorchData.DataLoader(dataset, shuffle=True, batch_size=bs * 2,
                                      num_workers=4, drop_last=False, pin_memory=True)
        target_loader_unl_list.append(loader)

    target_loader_val_list = []
    for dataset in target_dataset_val_list:
        loader = TorchData.DataLoader(dataset, shuffle=False, batch_size=bs * 2,
                                      num_workers=4, drop_last=False, pin_memory=True)
        target_loader_val_list.append(loader)

    target_loader_test_list = []
    for dataset in target_dataset_test_list:
        loader = TorchData.DataLoader(dataset, shuffle=False, batch_size=bs * 2,
                                      num_workers=4, drop_last=False, pin_memory=True)
        target_loader_test_list.append(loader)

    return source_loader, source_loader_copy, target_loader, target_loader_copy, target_loader_unl_list, \
        target_loader_val_list, target_loader_test_list


def build_dataset_sep(args):
    base_path = './data/txt/%s' % args.dataset
    root = args.data_root
    image_set_file_s = os.path.join(base_path, 'labeled_source_images_' + args.source + '.txt')

    image_set_file_t_list = [os.path.join(base_path, 'labeled_target_images_' + target + '_%d.txt' % (args.num)) for target in args.target_list]
    image_set_file_t_val_list = [os.path.join(base_path, 'validation_target_images_' + target + '_3.txt') for target in args.target_list]
    image_set_file_unl_list = [os.path.join(base_path, 'unlabeled_target_images_' + target + '_%d.txt' % (args.num)) for target in args.target_list]

    if args.domain_ind:
        domain_ind_list = [i for i in range(len(args.target_list))]
    else:
        domain_ind_list = [None] * len(args.target_list)

    if args.arch == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224

    data_transforms = get_transforms(crop_size)

    # load np array
    data_arr = np.load('/data/zizheng/office_home_image.npy')
    with open('/data/zizheng/office_home_index.pkl', 'rb') as f:
        index_dict = pickle.load(f)

    source_dataset = Imagelists_OfficeHome(image_set_file_s, data_arr, index_dict, root=root, transform=data_transforms['labeled'])

    # target labeled
    target_dataset_list = []
    for i, domain_ind in zip(image_set_file_t_list, domain_ind_list):
        dset = Imagelists_OfficeHome(i, data_arr, index_dict, root=root,
                                     transform=data_transforms['labeled'], domain_ind=domain_ind)
        target_dataset_list.append(dset)

    # target unlabeled
    target_dataset_unl_list = []
    for i, domain_ind in zip(image_set_file_unl_list, domain_ind_list):
        dset = Imagelists_OfficeHome(i, data_arr, index_dict, root=root,
                                     transform=data_transforms['fixmatch'], domain_ind=domain_ind)
        target_dataset_unl_list.append(dset)

    # target validation
    target_dataset_val_list = []
    for i in image_set_file_t_val_list:
        dset = Imagelists_OfficeHome(i, data_arr, index_dict, root=root, transform=data_transforms['val'])
        target_dataset_val_list.append(dset)

    # target test
    target_dataset_test_list = []
    for i in image_set_file_unl_list:
        dset = Imagelists_OfficeHome(i, data_arr, index_dict, root=root, transform=data_transforms['test'])
        target_dataset_test_list.append(dset)

    # target_dataset = TorchData.ConcatDataset(target_dataset_list)
    # target_dataset_unl = TorchData.ConcatDataset(target_dataset_unl_list)

    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    if args.arch == 'alexnet':
        bs = 32
    else:
        bs = 24

    train_sampler = TorchData.RandomSampler if args.local_rank == -1 else DistributedSampler

    source_loader = TorchData.DataLoader(source_dataset, sampler=train_sampler(source_dataset),
                                         batch_size=bs, num_workers=4, drop_last=True, pin_memory=True)

    target_loader_list = []
    for dataset in target_dataset_list:
        loader = TorchData.DataLoader(dataset, sampler=train_sampler(dataset),
                                      batch_size=min(bs, len(dataset)),
                                      num_workers=4, drop_last=True, pin_memory=True)
        target_loader_list.append(loader)

    target_loader_unl_list = []
    for dataset in target_dataset_unl_list:
        loader = TorchData.DataLoader(dataset, sampler=train_sampler(dataset),
                                      batch_size=int(bs * 2 / len(target_dataset_unl_list)), num_workers=4, drop_last=True, pin_memory=True)
        target_loader_unl_list.append(loader)

    target_loader_val_list = []
    for dataset in target_dataset_val_list:
        loader = TorchData.DataLoader(dataset, sampler=TorchData.SequentialSampler(dataset), batch_size=bs * 2,
                                      num_workers=4, drop_last=False, pin_memory=True)
        target_loader_val_list.append(loader)

    target_loader_test_list = []
    for dataset in target_dataset_test_list:
        loader = TorchData.DataLoader(dataset, sampler=TorchData.SequentialSampler(dataset), batch_size=bs * 2,
                                      num_workers=4, drop_last=False, pin_memory=True)
        target_loader_test_list.append(loader)

    return source_loader, target_loader_list, target_loader_unl_list, \
        target_loader_val_list, target_loader_test_list, class_list


class BuildDataDivide:
    def __init__(self, args, pseudo_path):
        self.args = args
        self.pseudo_path = pseudo_path
        base_path = './data/txt/%s' % args.dataset
        self.base_path = base_path
        root = args.data_root
        self.image_set_file_s = os.path.join(base_path, 'labeled_source_images_' + args.source + '.txt')

        self.image_set_file_t_list = [os.path.join(base_path, 'labeled_target_images_' + target + '_%d.txt' % (args.num)) for target in args.target_list]
        self.image_set_file_unl_list = [os.path.join(base_path, 'unlabeled_target_images_' + target + '_%d.txt' % (args.num)) for target in args.target_list]
        self.image_set_file_unl_pseudo_list = [os.path.join(pseudo_path, 'unlabeled_target_images_' + target + '_%d.txt' % (args.num)) for target in args.target_list]

        if args.arch == 'alexnet':
            self.crop_size = 227
            self.bs = 32
        else:
            self.crop_size = 224
            self.bs = 24

        # load np array
        self.data_arr = np.load('/data/zizheng/office_home_image.npy')
        with open('/data/zizheng/office_home_index.pkl', 'rb') as f:
            self.index_dict = pickle.load(f)

        self.clean_A = os.path.join(base_path, 'unlabeled_target_images_clean_Art_1.txt')
        self.noisy_A = os.path.join(base_path, 'unlabeled_target_images_noisy_Art_1.txt')

        self.clean_P = os.path.join(base_path, 'unlabeled_target_images_clean_Product_1.txt')
        self.noisy_P = os.path.join(base_path, 'unlabeled_target_images_noisy_Product_1.txt')

    def run_fixmatch(self):
        '''
        For single domain only
        '''
        root = self.args.data_root
        data_transforms = get_transforms(self.crop_size)
        assert len(self.args.target_list) == 1
        target_name = self.args.target_list[0]

        # target labeled
        path = self.image_set_file_t_list[0]
        gt_labeled_set = Imagelists_OfficeHome(path, self.data_arr, self.index_dict, root=root,
                                               transform=data_transforms['labeled'])

        path = os.path.join(self.base_path, 'unlabeled_target_images_clean_%s_%d.txt' % (target_name, self.args.num))
        pl_labeled_set = Imagelists_OfficeHome(path, self.data_arr, self.index_dict, root=root,
                                               transform=data_transforms['labeled'])

        path = os.path.join(self.base_path, 'unlabeled_target_images_noisy_%s_%d.txt' % (target_name, self.args.num))
        unlabeled_set = Imagelists_OfficeHome(path, self.data_arr, self.index_dict, root=root,
                                              transform=data_transforms['fixmatch'])

        labeled_set = TorchData.ConcatDataset([gt_labeled_set, pl_labeled_set])

        labeled_loader = TorchData.DataLoader(labeled_set, sampler=TorchData.RandomSampler(labeled_set),
                                              batch_size=min(self.bs, len(labeled_set)),
                                              num_workers=4, drop_last=True, pin_memory=True)
        unlabeled_loader = TorchData.DataLoader(unlabeled_set, sampler=TorchData.RandomSampler(unlabeled_set),
                                                batch_size=self.bs, num_workers=4, drop_last=True, pin_memory=True)

        return labeled_loader, unlabeled_loader

    def run(self, pred_list=None, prob_list=None, mode='train'):
        root = self.args.data_root
        data_transforms = get_transforms(self.crop_size)
        if mode == 'source':
            source_dataset = Imagelists_OfficeHome(self.image_set_file_s, self.data_arr, self.index_dict,
                                                   root=root, transform=data_transforms['labeled'])
            source_loader = TorchData.DataLoader(source_dataset, sampler=TorchData.RandomSampler(source_dataset),
                                                 batch_size=self.bs, num_workers=4, drop_last=True, pin_memory=True)
            return source_loader
        elif mode == 'warmup':
            # target labeled
            target_dataset_list = []
            for i in self.image_set_file_t_list:
                dset = Imagelists_OfficeHome(i, self.data_arr, self.index_dict, root=root,
                                             transform=data_transforms['labeled'])
                target_dataset_list.append(dset)

            # target pl
            target_dataset_pseudo_train_list = []
            for i in self.image_set_file_unl_pseudo_list:
                dset = Imagelists_OfficeHome(i, self.data_arr, self.index_dict, root=root,
                                             transform=data_transforms['labeled'])
                target_dataset_pseudo_train_list.append(dset)
            target_dataset_labeled = TorchData.ConcatDataset(target_dataset_list + target_dataset_pseudo_train_list)

            target_loader = TorchData.DataLoader(target_dataset_labeled, sampler=TorchData.RandomSampler(target_dataset_labeled),
                                                 batch_size=min(self.bs, len(target_dataset_labeled)),
                                                 num_workers=4, drop_last=True, pin_memory=True)
            return target_loader

        elif mode == 'train':
            # target labeled
            target_dataset_list = []
            for i in self.image_set_file_t_list:
                dset = Imagelists_OfficeHome_Divide(i, self.data_arr, self.index_dict, root=root,
                                                    transform=data_transforms['labeled'], mode='gt_labeled')
                target_dataset_list.append(dset)

            # target unl divide
            target_dataset_unl_labeled_list = []
            target_dataset_unl_unlabeled_list = []
            for domain_ind, i in enumerate(self.image_set_file_unl_pseudo_list):
                dset_labeled = Imagelists_OfficeHome_Divide(i, self.data_arr, self.index_dict, root=root,
                                                            transform=data_transforms['labeled'],
                                                            pred=pred_list[domain_ind], probability=prob_list[domain_ind].tolist(), mode='labeled')
                dset_unlabeled = Imagelists_OfficeHome_Divide(i, self.data_arr, self.index_dict, root=root,
                                                              transform=data_transforms['fixmatch'],
                                                              pred=pred_list[domain_ind], probability=prob_list[domain_ind].tolist(), mode='unlabeled')
                target_dataset_unl_labeled_list.append(dset_labeled)
                target_dataset_unl_unlabeled_list.append(dset_unlabeled)

            target_dataset_labeled = TorchData.ConcatDataset(target_dataset_list + target_dataset_unl_labeled_list)
            target_dataset_unlabeled = TorchData.ConcatDataset(target_dataset_unl_unlabeled_list)

            labeled_loader = TorchData.DataLoader(target_dataset_labeled, sampler=TorchData.RandomSampler(target_dataset_labeled),
                                                  batch_size=min(self.bs, len(target_dataset_labeled)),
                                                  num_workers=4, drop_last=True, pin_memory=True)
            unlabeled_loader = TorchData.DataLoader(target_dataset_unlabeled, sampler=TorchData.RandomSampler(target_dataset_unlabeled),
                                                    batch_size=self.bs, num_workers=4, drop_last=True, pin_memory=True)

            return labeled_loader, unlabeled_loader

        elif mode == 'test':
            # target test
            target_loader_test_list = []
            for i in self.image_set_file_unl_list:
                dset = Imagelists_OfficeHome(i, self.data_arr, self.index_dict, root=root, transform=data_transforms['test'])
                loader = TorchData.DataLoader(dset, sampler=TorchData.SequentialSampler(dset), batch_size=self.bs * 2,
                                              num_workers=4, drop_last=False, pin_memory=True)
                target_loader_test_list.append(loader)
            return target_loader_test_list

        elif mode == 'eval_train':
            # target test
            target_loader_test_list = []
            for i in self.image_set_file_unl_pseudo_list:
                dset = Imagelists_OfficeHome(i, self.data_arr, self.index_dict, root=root, transform=data_transforms['test'])
                loader = TorchData.DataLoader(dset, sampler=TorchData.SequentialSampler(dset), batch_size=self.bs * 2,
                                              num_workers=4, drop_last=False, pin_memory=True)
                target_loader_test_list.append(loader)
            return target_loader_test_list
