from torch import nn
import torch
import numpy as np
from torch.nn.functional import cross_entropy
from torch.optim import AdamW, Adam, SGD
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm.notebook import tqdm
import einops
from model import Transformer
import math

def create_data_p(p: int, function):
    x = torch.arange(p)  # 0..p
    y = torch.arange(p)  # 0..p
    x, y = torch.cartesian_prod(x, y).T  # декартово произведение x и y
    result = function(x, y) % p
    return torch.stack([x, y, result]).T

def prod(a, b):  # a*b
    return a * b

def calc_norm(model):
    return np.sqrt(sum(param.pow(2).sum().item() for param in model.parameters()))

def calc_si_norm(model):
    return np.sqrt(sum(param.pow(2).sum().item() if param.requires_grad else 0. for param in model.parameters()))

def calc_grad_norm(model):
    return np.sqrt(sum(param.grad.pow(2).sum().item() for param in model.parameters()))

def calc_norm_wot_last_layer(model):
    return np.sqrt(sum(param.grad.pow(2).sum().item() if not param.grad is None else 0. for param in model.parameters()))

def train(model):
    
    #craeating the dataset
    data = create_data_p(config['p'], config['func'])
    data = data.to(config['device'])
    data_index = torch.randperm(data.shape[0], device=config['device'])
    split = int(data.shape[0] * config['train_ratio'])
    training_set = data[data_index[:split]]
    validation_set = data[data_index[split:]]

    optimizer = config['opt'](net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    steps_per_epoch = math.ceil(training_set.shape[0] / config['batch_size'])
    train_acc, val_acc, train_loss, val_loss = [], [], [], []
    weights_norm, grad_norms = [], []
    norms = []
    effective_lr, effective_grad = [], []
    mean_effictive_grad = []
    mean_grad_norms = []
    attn1_norm, mlp1_norm = [], []
    attn1_grad_norm, mlp1_grad_norm = [], []
    attn2_norm, mlp2_norm = [], []
    attn2_grad_norm, mlp2_grad_norm = [], []
    mean_attn1_grad_norm, mean_mlp1_grad_norm = [], []
    mean_attn2_grad_norm, mean_mlp2_grad_norm = [], []

    k = 0
    for epoch in range(int(config['budget']) // steps_per_epoch):
        k += 1
        # на каждой эпохе перемешиваем train
        training_set = training_set[torch.randperm(training_set.shape[0]), :]

        for data, is_train in [(training_set, True), (validation_set, False)]:

            total_acc = 0
            total_loss = 0
            net.train(is_train)

            dl = torch.split(data, config['batch_size'], dim=0)  # делим на батчи
            for input in dl:  
                input = input.to(config['device'])  
                with torch.set_grad_enabled(is_train):
                    logits = net(input[:, :-1])  # предсказание
                    loss = cross_entropy(
                        logits, input[:, -1].flatten().to(torch.long))
                    total_loss += loss.item() * input.shape[0]

                if is_train:  # пересчитываем веса, вычисляя градиенты; обновляем lr
                    net.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    optimizer.step()

                    norm = calc_si_norm(net)
                    grad = calc_norm_wot_last_layer(net)
                    grad_norms.append(grad)
                    effective_grad.append(grad * norm)
                    weights_norm.append(norm)

                acc = (logits.argmax(-1) == input[:, -1]).float().mean()
                total_acc += acc.item() * input.shape[0]

            if is_train:
                train_acc.append(total_acc / training_set.shape[0])
                train_loss.append(total_loss / training_set.shape[0])
                norms.append(norm)

            else:
                val_acc.append(total_acc / validation_set.shape[0])
                val_loss.append(total_loss / validation_set.shape[0])

        effective_lr.append(optimizer.state_dict()['param_groups'][0]['lr'] / np.mean(weights_norm) ** 2)
        mean_effictive_grad.append(np.mean(effective_grad))
        mean_grad_norms.append(np.mean(grad_norms))

        effective_grad = []
        grad_norms = []
        weights_norm = []

        if train_acc[-1] == 1 and val_acc[-1] == 1:
            break
        print(f'Epoch {k}: Train / Val acc: {round(train_acc[-1], 4)} / {round(val_acc[-1], 4)}')

    plt.plot(train_acc, label='train')
    plt.plot(val_acc, label='val', alpha=0.7)
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('2L Transformer with SGD opt\n  lr=1e-1, weight_decay=1e-3')
    plt.grid()
    plt.savefig('grokking_2l_acc.pdf')

    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='val', alpha=0.7)
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('2L Transformer with SGD opt\n  lr=1e-1, weight_decay=1e-3')
    plt.grid()
    plt.savefig('grokking_2l_loss.pdf')

    plt.plot(mean_effictive_grad)
    plt.xlabel('epoch')
    plt.legend()
    plt.title('Mean effictive grad')
    plt.grid()
    plt.savefig('grokking_2l_grad.pdf')

    plt.plot(effective_lr)
    plt.xlabel('epoch')
    plt.legend()
    plt.title('Effictive learning rate')
    plt.grid()
    plt.savefig('grokking_2l_lr.pdf')

config = {
    'seed':14,
    'p': 97, 
    'device':  torch.device("cuda:0"), 
    'train_ratio': 0.4,
    'batch_size': 512, 
    'budget': 50000, 
    'func': prod, 
    'num_layers': 2, 
    'd_model': 128,
    'd_mlp': 512,
    'd_head': 32,
    'num_heads': 4,
    'n_ctx': 3,
    'act_type': 'ReLU',
    'use_cache': False,
    'use_ln': True,
    'opt': SGD,
    'lr': 1e-1,
    'weight_decay': 0.001
}

### main ####
np.random.seed(config['seed'])
torch.manual_seed(config['seed'])
torch.cuda.manual_seed(config['seed'])

net = Transformer(num_layers=config['num_layers'],
                    d_vocab=config['p'],
                    d_model=config['d_model'],
                    d_mlp=config['d_mlp'],
                    d_head=config['d_head'],
                    num_heads=config['num_heads'],
                    n_ctx=config['n_ctx'], # context length
                    act_type=config['act_type'],
                    use_cache=config['use_cache'],
                    use_ln=config['use_ln']
                 ).to(config['device'])
train(net)
