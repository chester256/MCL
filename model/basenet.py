from torchvision import models
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Function
import torchvision
import numpy as np


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


class RevGrad(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


grad_reverse = RevGrad.apply
# def grad_reverse(x, lambd=1.0):
#     return GradReverse(lambd)(x)


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))

    output = _output.view(input_size)

    return output


class AlexNetBase(nn.Module):
    def __init__(self, pret=True):
        super(AlexNetBase, self).__init__()
        model_alexnet = models.alexnet(pretrained=pret)
        self.features = nn.Sequential(*list(model_alexnet.
                                            features._modules.values())[:])
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i),
                                       model_alexnet.classifier[i])
        self.__in_features = model_alexnet.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self.__in_features

    def get_optim_params(self, lr):
        # lr_list, lr_multi_list = [], []
        # for name, param in self.named_parameters():
        #     if 'classifier' in name:
        #         lr_multi_list.append(param)
        #     else:
        #         lr_list.append(param)
        # return [{'params': lr_list, 'lr': lr},
        #         {'params': lr_multi_list, 'lr': 10 * lr}]
        lr_list, lr_multi_list = [], []
        for name, param in self.named_parameters():
            lr_list.append(param)
        return [{'params': lr_list, 'lr': lr},
                {'params': lr_multi_list, 'lr': 10 * lr}]


class VGGBase(nn.Module):
    def __init__(self, pret=True, no_pool=False):
        super(VGGBase, self).__init__()
        vgg16 = models.vgg16(pretrained=pret)
        self.classifier = nn.Sequential(*list(vgg16.classifier.
                                              _modules.values())[:-1])
        self.features = nn.Sequential(*list(vgg16.features.
                                            _modules.values())[:])
        self.s = nn.Parameter(torch.FloatTensor([10]))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 7 * 7 * 512)
        x = self.classifier(x)
        return x


class Predictor(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Predictor, self).__init__()
        self.fc = nn.Linear(inc, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        if reverse:
            x = grad_reverse(x, torch.tensor(eta, requires_grad=False))
        x = F.normalize(x)
        x_out = self.fc(x) / self.temp
        return x, x_out

    def get_optim_params(self, lr):
        lr_multi_list = []
        for name, param in self.named_parameters():
            lr_multi_list.append(param)
        return [{'params': lr_multi_list, 'lr': 10 * lr}]


class Predictor_deep(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Predictor_deep, self).__init__()
        self.fc1 = nn.Linear(inc, 512)
        self.fc2 = nn.Linear(512, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x, torch.tensor(eta, requires_grad=False))
        x = F.normalize(x)
        x_out = self.fc2(x) / self.temp
        return x, x_out

    def get_optim_params(self, lr):
        lr_multi_list = []
        for name, param in self.named_parameters():
            lr_multi_list.append(param)
        return [{'params': lr_multi_list, 'lr': 10 * lr}]


class Predictor_cos(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super().__init__()
        self.fc1 = nn.Linear(inc, 512)
        # self.fc2 = nn.Linear(512, num_class, bias=False)
        self.fc2 = nn.Parameter(torch.randn(num_class, 512))
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x, torch.tensor(eta, requires_grad=False))
        x = F.normalize(x)

        # self.fc2.data = F.normalize(self.fc2.data)
        # x_out = x @ self.fc2.t() / self.temp

        fc2_norm = F.normalize(self.fc2)
        x_out = x @ fc2_norm.t() / self.temp
        return x_out

    def get_optim_params(self, lr):
        lr_multi_list = []
        for param in self.parameters():
            lr_multi_list.append(param)
        return [{'params': lr_multi_list, 'lr': 10 * lr}]


class Predictor_cos_feat(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super().__init__()
        self.fc1 = nn.Linear(inc, 512)
        # self.fc2 = nn.Linear(512, num_class, bias=False)
        self.fc2 = nn.Parameter(torch.randn(num_class, 512))
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x, torch.tensor(eta, requires_grad=False))
        x = F.normalize(x)

        fc2_norm = F.normalize(self.fc2)
        x_out = x @ fc2_norm.t() / self.temp
        return x, x_out

    def get_optim_params(self, lr):
        lr_multi_list = []
        for param in self.parameters():
            lr_multi_list.append(param)
        return [{'params': lr_multi_list, 'lr': 10 * lr}]


class PredictorTSA(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(inc, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(512, num_class)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=1):
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x, torch.tensor(eta, requires_grad=False))
        y = self.fc2(x)
        return x, y

    def get_optim_params(self, lr):
        lr_multi_list = []
        for param in self.parameters():
            lr_multi_list.append(param)
        return [{'params': lr_multi_list, 'lr': 10 * lr}]


class PredictorTri(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(inc, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout()
        )
        self.fc3 = nn.Linear(256, num_class)
        self.num_class = num_class

    def forward(self, x, reverse=False, eta=0.1):
        x = self.fc1(x)
        x = self.fc2(x)
        y = self.fc3(x)
        return x, y

    def get_optim_params(self, lr):
        lr_multi_list = []
        for param in self.parameters():
            lr_multi_list.append(param)
        return [{'params': lr_multi_list, 'lr': 10 * lr}]


class ResBase34(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = torchvision.models.resnet18(pretrained=True)
        self.conv1 = self.model.conv1
        self.bn1 = self.model.bn1
        self.maxpool = self.model.maxpool
        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4
        self.avgpool = self.model.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def get_optim_params(self, lr):
        lr_list, lr_multi_list = [], []
        for name, param in self.named_parameters():
            if 'fc' in name:
                lr_multi_list.append(param)
            else:
                lr_list.append(param)
        return [{'params': lr_list, 'lr': lr},
                {'params': lr_multi_list, 'lr': 10 * lr}]


class Discriminator(nn.Module):
    def __init__(self, inc=4096):
        super(Discriminator, self).__init__()
        self.fc1_1 = nn.Linear(inc, 512)
        self.fc2_1 = nn.Linear(512, 512)
        self.fc3_1 = nn.Linear(512, 1)

    def forward(self, x, reverse=True, eta=1.0):
        if reverse:
            x = grad_reverse(x, torch.tensor(eta, requires_grad=False))
        x = F.relu(self.fc1_1(x))
        x = F.relu(self.fc2_1(x))
        x_out = F.sigmoid(self.fc3_1(x))
        return x_out


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y


if __name__ == '__main__':
    model = AlexNetBase(pret=False)
    for name, param in model.named_parameters():
        print(name)
