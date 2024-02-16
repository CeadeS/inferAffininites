import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
import torch
import os, shutil
import seaborn as sns
from torchvision.transforms import Resize, ToPILImage
from torchvision import transforms
from torch.nn import Module
from torch.jit import script
from torch import nn
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss
from pytorch_h5dataset.dataset.imageDataset import ImageDataset
from pytorch_h5dataset.dataloader.dataLoader import DataLoader
from torchvision.models import resnet50
from skorch.callbacks import EpochScoring, BatchScoring
from sklearn.metrics import accuracy_score
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import hydra
import yaml
from yamlable import YamlAble, yaml_info
from functools import partial
import lr_scheduler

@hydra.main(version_base=None, config_path="./config", config_name="base.yaml")
def main(cfg):
    print(cfg)
    def tf_norm(im):
        return (im-0.5)/0.5
    assert cfg.exp.selector in ['naive', 'species', 'genus', 'family','order', 'subclass']
    assert all(t in ['all', 'species', 'genus', 'family','order', 'subclass'] for t in cfg.exp.target)
    assert cfg.exp.mode in ['class', 'dist_fit', 'all']
    if cfg.exp.target == ['all']:
        cfg.exp.target = ['species', 'genus', 'family','order', 'subclass']
    sel = cfg.exp.selector
    cfg.optim.use_amp = False if cfg.exp.device == 'cpu' else cfg.optim.use_amp 
    
    scheduler = lr_scheduler.CosineAnnealingWithWarmup(
                                            total_epochs= cfg.optim.epochs,
                                            warmup_epochs = cfg.optim.lr_warmup,
                                            min_lr = 0,
                                            last_epoch= -1)

    

    train_augment = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.autoaugment.RandAugment(),
            transforms.autoaugment.AugMix(),
            transforms.ToTensor(),
            tf_norm,
            ])


    train_dataset = ImageDataset(f'muscheln_{sel}_v1', f'../data/muscheln/h5/{sel}_train',
                     split_mode = 'full',
                     tr_crop_strategy = 'random',
                     #tr_crop_size = (0.733,1.3),
                     #tr_crop_area_ratio_range = (.7,1.0),
                     tr_crop_size = (0.9,1.1),
                     tr_crop_area_ratio_range = (.95,1.0),                                 
                     tr_output_size = (244, 244),
                     decode = 'cpu',# 'cuda', # cpu, cuda
                     output_device=  'cpu', #cpu or cuda
                     tensor_transforms = train_augment,
                     quality=83)
    edge_lengths = pd.read_csv('meta/edge_lengths.csv')
    
    train_loader = DataLoader(dataset=train_dataset,
                 device='cpu', 
                 return_meta_indices=False,
                 pin_memory=True, 
                 batch_size=cfg.data.batch_size, 
                 num_workers= cfg.data.num_worker, ## 32 runs 700 * 100 samples /second
                 shuffle=True,
                 sampler=None,
                 num_batches_buffered=1,
                persistent_workers=False,
                multiprocessing_context='fork',
                meta_filter=None)
    
    test_augment = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(size=(244,244)),
            transforms.ToTensor(),
            tf_norm,
            ])


    test_dataset = ImageDataset(f'muscheln_{sel}_v1', f'../data/muscheln/h5/{sel}_test',
                     split_mode = 'full',
                     tr_crop_strategy = 'center',
                     #tr_crop_size = 1.0,
                     #tr_crop_area_ratio_range = (0.95),
                     tr_crop_size = 1.0,
                     tr_crop_area_ratio_range = (1.0),
                     tr_output_size = (244, 244),                                
                     decode = 'cpu',# 'cuda', # cpu, cuda
                     output_device=  'cpu', #cpu or cuda
                     tensor_transforms = test_augment,
                     quality=83)
    edge_lengths = pd.read_csv('meta/edge_lengths.csv')
    
    test_loader = DataLoader(dataset=test_dataset,
                 device='cpu', 
                 return_meta_indices=False,
                 pin_memory=True, 
                 batch_size=cfg.data.batch_size, 
                 num_workers= cfg.data.num_worker, ## 32 runs 700 * 100 samples /second
                 shuffle=False,
                 sampler=None,
                 num_batches_buffered=1,
                 #normalize=normalization,
                persistent_workers=False,
                multiprocessing_context='fork',
                meta_filter=None)

    class SmoothCrossEntropyLoss(_WeightedLoss):
        def __init__(self, weight=None, reduction='mean', smoothing=0.0):
            super().__init__(weight=weight, reduction=reduction)
            self.smoothing = smoothing
            self.weight = weight
            self.reduction = reduction        

        def k_one_hot(self, targets:torch.Tensor, n_classes:int, smoothing=0.0):
            with torch.no_grad():
                targets = torch.empty(size=(targets.size(0), n_classes),
                                      device=targets.device) \
                                      .fill_(smoothing /(n_classes-1)) \
                                      .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
            return targets

        def reduce_loss(self, loss):
            return loss.mean() if self.reduction == 'mean' else loss.sum() \
            if self.reduction == 'sum' else loss

        def forward(self, inputs, targets):
            assert 0 <= self.smoothing < 1

            targets = self.k_one_hot(targets, inputs.size(-1), self.smoothing)
            log_preds = F.log_softmax(inputs, -1)

            if self.weight is not None:
                log_preds = log_preds * self.weight.unsqueeze(0)

            return self.reduce_loss(-(targets * log_preds).sum(dim=-1))


    class CrossEntropyLoss(_WeightedLoss):
        def __init__(self, weight=None, reduction='mean'):
            super().__init__(weight=weight, reduction=reduction)
            self.weight = weight
            self.reduction = reduction

        def reduce_loss(self, loss):
            return loss.mean() if self.reduction == 'mean' else loss.sum() if self.reduction == 'sum' else loss

        def forward(self, inputs, targets):

            log_preds = F.log_softmax(inputs, -1)

            if self.weight is not None:
                log_preds = log_preds * self.weight.unsqueeze(0)

            return self.reduce_loss(-(targets * log_preds).sum(dim=-1))
    cl_loss = torch.jit.script(CrossEntropyLoss())
    cl_loss = torch.jit.script(torch.nn.CrossEntropyLoss())
    ce_loss = torch.jit.script(torch.nn.CrossEntropyLoss())

    from torchvision import models as models, datasets
    from classifiers import Simple, Parallel
    #(4325, 949, 78, 27, 5)
    num_species, num_genera, num_families, num_orders, num_sub_classes = (4325+1, 949+1, 78+1, 27+1, 5+1)
    model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
    cl_weight =  model.fc.weight.data.detach()


    
    if cfg.model.name == 'simple':
        print("Sequential Model")
        model.fc = Simple(cl_weight, num_species, num_genera, num_families, num_orders, num_sub_classes)
    elif cfg.model.name == 'parallel':
        print("Parallel Model")
        model.fc = Parallel(cl_weight, num_species, num_genera, num_families, num_orders, num_sub_classes)
    else:
        raise NotImplementedError()

    batch_size=cfg.data.batch_size
    cl_loss = torch.jit.script(CrossEntropyLoss())

    i = 0
    
    conv_parameters= []
    for name, param in model.named_parameters():
        if 'fc' not in name:
            conv_parameters.append(param)

    optimizer = torch.optim.Adam( 
        [
            {'params': iter(conv_parameters), 'lr': cfg.optim.model.features.lr},
            {'params': model.fc.parameters(), 'lr': cfg.optim.model.classifier.lr}
        ],cfg.optim.model.lr)
    scheduler = scheduler(optimizer)
    from time import time
    
    t0 = time()
    history_history= []
    test_history_history= []
    scaler = GradScaler(enabled = cfg.optim.use_amp)
    ce_loss= torch.nn.CrossEntropyLoss()
    model = model.to(cfg.exp.device)
    
    class MultiLoss(nn.Module):
        def __init__(self, mode, target, edge_lengths=None):
            super(MultiLoss, self).__init__()
            self.mode = mode
            self.target = target
            self.ce_loss = torch.jit.script(torch.nn.CrossEntropyLoss())
            self.chan_map = dict(zip(('species','genus','family','order','subclass'),range(5)))           
                            
            def __class_loss__(logit_tuple, meta):
                loss = 0.
                for t in target:
                    x = logit_tuple[self.chan_map[t]]
                    y = torch.LongTensor(meta[f'{t}_idx'].values.flatten()).to(cfg.exp.device)
                    loss += self.ce_loss(x,y)
                return loss
                                    
            self.edge_lengths = edge_lengths
            def __gen_dist_loss__(x, meta):
                targets = torch.stack([torch.FloatTensor(edge_lengths[edge_lengths.index==fam].drop(columns='family').values.flatten()) for fam in meta['family_idx']]).to(cfg.exp.device)
                soft_targets = F.softmax(-targets,dim=-1)
                return self.ce_loss(x[2], soft_targets)
            
            if mode == 'all':
                def __func__(x, meta):
                    loss = __class_loss__(x, meta)                    
                    loss += __gen_dist_loss__(x, meta)
                    return loss
            elif mode == 'dist_fit':
                def __func__(x, meta):
                    loss = __gen_dist_loss__(x, meta)      
                    return loss
            else:
                __func__ = __class_loss__
            self.forward = __func__
            
    class MultiAcc(nn.Module):
        def __init__(self, target, epoch=0):
            super(MultiAcc, self).__init__()
            self.target = target
            self.chan_map = dict(zip(('species','genus','family','order','subclass'),range(5)))  
            self.epoch = epoch
            self._eval = False
        
        def eval(self):
            self._eval = True
            
        def train(self):
            self._eval = False
                            
        def forward(self, logit_tuple, meta):
            loss = 0.
            acc = []
            if self._eval:
                Y, _Y, idxs, T, EPOCH = [], [], [], [], []
                for t in self.target:
                    x = logit_tuple[self.chan_map[t]]
                    y = torch.LongTensor(meta[f'{t}_idx'].values.flatten())
                    _y = x.argmax(dim=-1).detach().cpu().numpy()
                    acc.append(accuracy_score(_y, y))
                    Y.extend(y.tolist())
                    _Y.extend(_y.tolist())
                    T.extend(list(t for _ in range(len(y))))
                    EPOCH.extend(list(self.epoch for _ in range(len(y))))
                    idxs.extend(list(meta['Index']))
                accd = dict(zip(self.target,acc))
                accd['mean'] = float(np.array(acc).mean())
                accd['std'] = float(np.array(acc).std())
                return accd, {'Y_gt': Y, 'Y_pred':_Y, 'Index': idxs, 'T':T, 'Epoch':EPOCH}
            else:
                for t in self.target:
                    x = logit_tuple[self.chan_map[t]]
                    y = torch.LongTensor(meta[f'{t}_idx'].values.flatten())
                    _y = x.argmax(dim=-1).detach().cpu().numpy()
                    acc.append(accuracy_score(_y, y))
                accd = dict(zip(self.target,acc))
                accd['mean'] = float(np.array(acc).mean())
                accd['std'] = float(np.array(acc).std())
                return accd
    
    loss_func = MultiLoss(cfg.exp.mode, cfg.exp.target, edge_lengths)
    acc_score = MultiAcc(cfg.exp.target)
    RES = []# pd.DataFrame(columns= ['Index', 'Y_gt', 'Y_pred'])
    best_acc = 0
    if True:
        torch.save(model, "compet_model.tar.gz")
    with tqdm(total=cfg.optim.epochs) as pbar:
        
        for epoch in range(cfg.optim.epochs):
            acc_score.epoch=epoch
            times = [0.]
            means= []
            acc_score.train()
            for i, (im, (label, meta)) in enumerate(train_loader):
               
                im, label= im.to(cfg.exp.device), label.to(cfg.exp.device)
            
                with autocast(cfg.optim.use_amp):
                    x = model(im)
                    loss = loss_func(x, meta)
                    accd  = acc_score(x,meta)
                    means.append(accd['mean'])
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                i+=1
                pbar.set_postfix({'mode:': 'train',
                                  'epoch': f"{epoch}", 
                                  'step':f"{i}", 
                                  's/ep':f"{len(train_loader)}",
                                  'acc':f"{np.mean(np.array(means))*100:3.2f}",
                                  'loss':f"{loss:3.4f}"
                                 })

            scheduler.step()            
            with torch.no_grad():
                acc_score.eval()
                means = []
                for i, (im, (label, meta)) in enumerate(test_loader):
                    im, label= im.to(cfg.exp.device), label.to(cfg.exp.device)
            
                    x = model(im)
                    loss = loss_func(x, meta)
                    accd, res = acc_score(x,meta)
                    means.append(accd['mean'])
                    RES.append(pd.DataFrame(res))
                    i+=1
                    pbar.set_postfix({'mode:': 'test',
                                      'epoch': f"{epoch}", 
                                      'step':f"{i}", 
                                      's/ep':f"{len(test_loader)}",
                                      'acc':f"{np.mean(np.array(means))*100:3.2f}",
                                      'loss':f"{loss:3.4f}"
                                     })
            if True:
                if np.mean(np.array(means)) > best_acc:
                    torch.save(model, "compet_model.tar.gz")
                    best_acc = np.mean(np.array(means))
            out_res = pd.concat(RES)
            out_res.to_csv(f'result_{cfg.exp.name}.csv')                          
            pbar.update(1)
    

if __name__ == "__main__" :
    main()
