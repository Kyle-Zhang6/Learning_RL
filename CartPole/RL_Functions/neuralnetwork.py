from torch import nn, optim
from collections import OrderedDict


def set_freeze_all(model, freeze=True):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = not freeze


class MLP(nn.Module):
    def __init__(self, kwargs):
        super(MLP, self).__init__()
        keys = kwargs.keys()

        feature_list = [kwargs['input_feature']] + kwargs['hidden_features'] + [kwargs['output_feature']]
        layers = []
        if len(feature_list) > 1:
            for i in range(1, len(feature_list) - 1):
                layers.append(nn.Linear(feature_list[i - 1], feature_list[i]))
                layers.append(nn.ReLU())
        layers.append(nn.Linear(feature_list[-2], feature_list[-1]))
        if 'softmax' in keys:
            if kwargs['softmax']:
                layers.append(nn.Softmax(dim=-1))

        layer_names = [('layer' + str(i)) for i in list(range(len(layers)))]
        layers = dict(zip(layer_names, layers))
        self.layers = nn.Sequential(OrderedDict(layers))

        lr = 0.01
        if 'lr' in keys:
            lr = kwargs['lr']
        # Initiate optimizer
        if 'optimizer' in kwargs.keys():
            self.optimizer = kwargs['optimizer'](self.parameters(), lr=lr)
        else:
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # Initiate scheduler
        self.scheduler = kwargs['scheduler'](self.optimizer, **kwargs['scheduler_parameters']) \
            if 'scheduler' in keys else None
        # Initiate criterion
        self.criterion = kwargs['criterion'] if 'criterion' in keys else None

    def forward(self, x):
        x = self.layers(x)
        return x
