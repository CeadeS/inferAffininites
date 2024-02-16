import torch
import torchvision.models as models
import numpy as np
import os
import torch
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
from torch.multiprocessing import cpu_count
import torchvision.transforms as T
from flash.core.optimizers import lars
from torch.distributed import get_world_size

import logging

logger = logging.getLogger(__name__)
#logger.setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR, format='%(message)s')

try:
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    #print(get_world_size())
except:
    local_rank, global_rank = None, None


# a lazy way to pass the config file
class Hparams:
    def __init__(self):
        self.epochs = 1000 # number of training epochs
        self.seed = 77777 # randomness seed
        self.cuda = True # use nvidia gpu
        #self.img_size = 96 #image shape
        self.save = "./saved_models/" # save checkpoint
        self.load = False # load pretrained checkpoint
        self.gradient_accumulation_steps = 1 # gradient accumulation steps
        self.batch_size = 56*8
        self.img_size = 176 ## https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/ # https://github.com/pytorch/vision/tree/main/references/classification
        self.lr = 1.0#3e-4 # for ADAm only
        self.weight_decay = 1e-6
        self.embedding_size= 128 # papers value is 128
        self.temperature = 0.5 # 0.1 or 0.5
        self.checkpoint_path = './SimCLR_ResNet18_adam_.ckpt' # replace checkpoint path here

def default(val, def_val):
    return def_val if val is None else val

def reproducibility(config):
    SEED = int(config.seed)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    if (config.cuda):
        torch.cuda.manual_seed(SEED)


def device_as(t1, t2):
    """
    Moves t1 to the device of t2
    """
    return t1.to(t2.device)

# From https://github.com/PyTorchLightning/pytorch-lightning/issues/924
def weights_update(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f'Checkpoint {checkpoint_path} was loaded')
    return model

def tensor_to_rgb(t: torch.tensor):
    #print(t.min(), t.max())
    if t.shape[0]!=3:
        t = t.repeat(3, 1, 1)
    return t

import numpy as np
def bbb(bb):
    print(np.array(bb).min(), np.array(bb).max())
    return bb


class Augment:
    """
    A stochastic data augmentation module
    Transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, img_size, s=1):
        color_jitter = T.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        # 10% of the image
        blur = T.GaussianBlur((3, 3), (0.1, 2.0))

        self.train_transform = T.Compose(
            [
            T.RandomResizedCrop(size=img_size),
            T.RandomHorizontalFlip(p=0.5),  # with 0.5 probability
            T.RandomApply([color_jitter], p=0.8),
            T.RandomApply([blur], p=0.5),
            T.RandomGrayscale(p=0.2),
            # imagenet stats
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

        self.test_transform = T.Compose(
            [
             #tensor_to_rgb,   
                #T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, x):
        #print(type(x))
        #return self.train_transform(x), self.train_transform(x)
        return self.train_transform(x), self.train_transform(x)



def get_stl_dataloader(batch_size, transform=None, split="unlabeled"):
    raise NotImplementedError

    stl10 = STL10("./", split=split, transform=transform, download=True)
    return DataLoader(dataset=stl10, batch_size=batch_size, num_workers=cpu_count()//2)

import matplotlib.pyplot as plt

def imshow(img):
    """
    shows an imagenet-normalized image on the screen
    """
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    npimg = unnormalize(img).numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



class ContrastiveLoss(nn.Module):
    """
    Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
    """
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()

    def calc_similarity_batch(self, a, b):
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    def forward(self, proj_1, proj_2):
        """
        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
        where corresponding indices are pairs
        z_i, z_j in the SimCLR paper
        """
        batch_size = proj_1.shape[0]
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)

        denominator = device_as(self.mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * self.batch_size)
        return loss

import torchvision.transforms as transforms
from zipdataset import ImageZipDataset

def get_stl_dataloader(batch_size, transform=None, split="train"):
    if split == 'unlabelled' or not split=='test':
        split = 'train'
        
    root = '/home/mhofmann/nas/home/py_projects/data/muscheln'
sel = "naive"
split='train'
import torchvision.transforms as transforms
from zipdataset import ImageZipDataset
import pandas as pd

def load_pickle(name, delimiter):
            return pd.read_pickle(name[:-4]+'.pkl')
    
def get_muschel_dataloader(batch_size, transform=None, split="train"):
    if split == 'unlabelled' or not split=='test':
        split = 'train'
        
    dataset =  ImageZipDataset("../IMAGES_final.zip", "/home/mhofmann/nas/home/py_projects/muscheln/zero_shot/meta.pkl",
                                    transform=transform,
                                    load_fn=load_pickle,
                                    split='train',
                                    l_key="species_idx",
                                    f_key="FilePath",
                                    eval_proportion=0.00)
    
    data_loader = DataLoader(dataset=dataset,
                 pin_memory=True, 
                 batch_size=batch_size, 
                 num_workers= cpu_count()//16,
                 shuffle=True,
                 sampler=None,
                 drop_last=True)

    return data_loader


import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim import SGD, Adam


class AddProjection(nn.Module):
    def __init__(self, config, model=None, mlp_dim=512):
        super(AddProjection, self).__init__()
        embedding_size = config.embedding_size
        self.backbone = default(model, models.resnet18(pretrained=False, num_classes=config.embedding_size))
        mlp_dim = default(mlp_dim, self.backbone.fc.in_features)
        print('Dim MLP input:',mlp_dim)
        self.backbone.fc = nn.Identity()

        # add mlp projection head
        self.projection = nn.Sequential(
            nn.Linear(in_features=mlp_dim, out_features=mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(),
            nn.Linear(in_features=mlp_dim, out_features=embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

    def forward(self, x, return_embedding=False):
        embedding = self.backbone(x)
        if return_embedding:
            return embedding
        return self.projection(embedding)



def define_param_groups(model, weight_decay, optimizer_name):
    def exclude_from_wd_and_adaptation(name):
        if 'bn' in name:
            return True
        if optimizer_name == 'lars' and 'bias' in name:
            return True

    param_groups = [
        {
            'params': [p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)],
            'weight_decay': weight_decay,
            'layer_adaptation': True,
        },
        {
            'params': [p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)],
            'weight_decay': 0.,
            'layer_adaptation': False,
        },
    ]
    return param_groups


class SimCLR_pl(pl.LightningModule):
    def __init__(self, config, model=None, feat_dim=512):
        super().__init__()
        self.config = config
        
        self.model = AddProjection(config, model=model, mlp_dim=feat_dim)

        self.loss = ContrastiveLoss(config.batch_size, temperature=self.config.temperature)

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_idx):
        
        #(imstack), labels = batch
        #x1,x2 = imstack[:,0,...],imstack[:,1,...]
        (x1,x2) , labels = batch
        z1 = self.model(x1)
        z2 = self.model(x2)
        loss = self.loss(z1, z2)
        self.log('Contrastive loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        max_epochs = int(self.config.epochs)
        param_groups = define_param_groups(self.model, self.config.weight_decay, 'adam')
        lr = self.config.lr
        #optimizer = Adam(param_groups, lr=lr, weight_decay=self.config.weight_decay)
        optimizer = lars.LARS(param_groups, lr=lr, weight_decay=self.config.weight_decay)
        if global_rank==0:
            print(f'Optimizer Lars, '
              f'Learning Rate {lr}, '
              f'Effective batch size {self.config.batch_size * self.config.gradient_accumulation_steps * world_size}')

        scheduler_warmup = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=max_epochs,
                                                         warmup_start_lr=0.0)

        return [optimizer], [scheduler_warmup] 
    

if __name__ == "__main__":
    import torch
    from pytorch_lightning import Trainer
    import os
    from pytorch_lightning.callbacks import GradientAccumulationScheduler
    from pytorch_lightning.callbacks import ModelCheckpoint
    from torchvision.models import  resnet18
    import torchvision

    available_gpus = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
    save_model_path = os.path.join(os.getcwd(), "saved_models/")
    print('available_gpus:',available_gpus)
    filename=f'Muscheln_resnet_v2_rank_{global_rank}'
    resume_from_checkpoint = False
    train_config = Hparams()
    train_config.lr = 0.3 * train_config.batch_size*world_size * train_config.gradient_accumulation_steps/256
    if global_rank==0:
        print(f"Effective Learning Rate: {train_config.lr}")
        print(f"Effective BatchSize {train_config.batch_size*world_size * train_config.gradient_accumulation_steps}")

    reproducibility(train_config)
    save_name = filename + '.ckpt'

    #model = SimCLR_pl(train_config, model=resnet18(pretrained=False), feat_dim=512)

    m = torchvision.models.get_model("resnet50", num_classes=5)
    model = SimCLR_pl(train_config, model=m, feat_dim=2048)


    transform = Augment(train_config.img_size)
    data_loader = get_muschel_dataloader(train_config.batch_size, transform)

    accumulator = GradientAccumulationScheduler(scheduling={0: train_config.gradient_accumulation_steps})

        
    checkpoint_callback =  ModelCheckpoint(filename=filename, dirpath=save_model_path,every_n_epochs=25,
                                            save_last=True, save_top_k=2,monitor='Contrastive loss_epoch',mode='min')

    if resume_from_checkpoint:
        print("##############")
        print("Resuming from Checkpoint")
        print("##############")
        trainer = Trainer(callbacks=[accumulator, checkpoint_callback],
                      gpus=available_gpus, #limit_train_batches=5,
                      max_epochs=train_config.epochs, strategy='ddp', num_nodes=3,
                      resume_from_checkpoint=train_config.checkpoint_path)
    else:
        trainer = Trainer(callbacks=[accumulator, checkpoint_callback],
                      gpus=available_gpus, strategy='ddp', num_nodes=3, #limit_train_batches=5,
                      max_epochs=train_config.epochs)


    trainer.fit(model, data_loader)

    if global_rank == 0:
        trainer.save_checkpoint(save_name)