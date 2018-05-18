import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import numpy as np
import sys
import os



class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.optimizer = optimizer
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + np.cos(np.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]
    
    def _reset(self, epoch, T_max):
        """
        Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        return CosineAnnealingLR(self.optimizer, self.T_max, self.eta_min, last_epoch=epoch)


    
def seq2seq_loss(input, target):
    sl, bs = target.size()
    sl_in, bs_in, nc = input.size()
    if sl > sl_in: 
        input = F.pad(input, (0,0,0,0,0,sl-sl_in))
    input = input[:sl]
    return F.cross_entropy(input.view(-1,nc), target.view(-1))

def trainable_params_(m):
    return [p for p in m.parameters() if p.requires_grad]


from tqdm import tqdm_notebook, tnrange, tqdm

def format_tensor(X):
    X = X.transpose(0,1)
    return Variable(X.long()).cuda()


def fit(model, train_loader, opt_fn=None, learning_rate=1e-5, epochs=1, cycle_len=1, val_loader=None, metrics=None, 
                save=False, save_path='models/checkpoint.pth.tar', pre_saved=False, print_period=100, grad_clip=0.0, 
                tr_ratios=None, pad=1, return_history=False):
       
    if tr_ratios is None:
        tr_ratios = np.linspace(1.0, 0.0, num=epochs)
    assert(len(tr_ratios) == epochs), 'need to have same len of "tr_ratios" as number of epochs'
        
    if opt_fn:
        optimizer = opt_fn(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    else:  
        optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    # for stepper 
    n_batches = int(len(train_loader.dataset) // train_loader.batch_size)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_batches*cycle_len)
    global all_lr
    all_lr = []
    all_train_loss = []
    all_valid_loss = []
    
    best_val_loss = np.inf
    
    if pre_saved:
        checkpoint = torch.load(save_path)
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('...restoring model...')
    begin = True
    
    total_loss = 0.0
    num_print_periods = 0
    for epoch_ in tnrange(1, epochs+1, desc='Epoch'):
        
        if pre_saved:      
            if begin:
                epoch = epoch_
                begin = False
        else:
            epoch = epoch_
        
        # training
        train_loss = train(model, train_loader, optimizer, scheduler, tr_ratios, epoch_-1, grad_clip, 
                           print_period, total_loss, num_print_periods)
        all_train_loss.append(train_loss)
        print_output = [epoch, train_loss]
        # validation
        if val_loader:
            val_loss = validate(model, val_loader, optimizer)
            all_valid_loss.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                # save model     
                if save:
                    if save_path:
                        ensure_dir(save_path)
                        state = {
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'best_val_loss': best_val_loss,
                            'optimizer': optimizer.state_dict()
                        }
                        save_checkpoint(state, save_path=save_path)
            
            print_output.append(val_loss)

        print('\n', print_output)

        # reset scheduler
        if epoch_ % cycle_len == 0:
            scheduler = scheduler._reset(epoch, T_max=n_batches*cycle_len)
        
        epoch += 1
        total_loss = 0.0
        num_print_periods = 0
        
    history = {
        'all_lr': all_lr,
        'train_loss': all_train_loss,
        'val_loss': all_valid_loss,
    }

    if return_history:
        return history
        
    
def train(model, train_loader, optimizer, scheduler, tr_ratios, epoch_, grad_clip=0.0, print_period=1000, 
            total_loss=0, num_print_periods=0):

    epoch_loss = 0.
    n_batches = int(len(train_loader.dataset) / train_loader.batch_size)
    model.train()
    
    for i, (text, summary) in enumerate(train_loader):
        optimizer.zero_grad()
        text = format_tensor(text)
        summary = format_tensor(summary)
                                  
        output = model(text, summary, tr_ratios[epoch_])
        l = seq2seq_loss(output, summary)
        epoch_loss += l.data[0]
        l.backward()
        optimizer.step()
        scheduler.step()
        all_lr.append(scheduler.get_lr())
        
        clip_grad_norm(trainable_params_(model), grad_clip)
        if i % print_period == 0 and i != 0:
            epoch_loss = epoch_loss / print_period
            statement = 'epoch_loss: {:.5f}, % of epoch: ({:.0f}%)'.format(epoch_loss, (i / n_batches)*100.)
            sys.stdout.write('\r' + statement)
            sys.stdout.flush()
            total_loss += epoch_loss
            epoch_loss = 0.0   
            num_print_periods += 1
#     raise ZeroDivisionError('Not enough data for default "print_period", please lower "print_period"')
    train_loss = total_loss / num_print_periods
    return train_loss


def validate(model, val_loader, optimizer):
    model.eval()
    n_batches = int(len(val_loader.dataset) / val_loader.batch_size)
    total_loss = 0.0

    for i, (text, summary) in enumerate(val_loader):
        text = format_tensor(text)
        summary = format_tensor(summary)
        output = model(text)
        l = seq2seq_loss(output, summary)
        total_loss += l.data[0]
        
    return total_loss / n_batches    


def save_checkpoint(state, save_path='models/checkpoint.pth.tar'):
    torch.save(state, save_path)
    
def save_model(model, save_path='models/checkpoint.pth.tar'):
    model.save_state_dict(save_path)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)