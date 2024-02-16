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
from muschel_dataset import *

if __name__ == '__main__':    

    import torch
    from pytorch_lightning import Trainer
    import os
    from pytorch_lightning.callbacks import GradientAccumulationScheduler
    from pytorch_lightning.callbacks import ModelCheckpoint
    from torchvision.models import  resnet18
    

    available_gpus = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
    save_model_path = os.path.join(os.getcwd(), "saved_models/")
    print('available_gpus:',available_gpus)
    filename='SimCLR_ResNet18_adam_'
    resume_from_checkpoint = False
    train_config = Hparams()

    reproducibility(train_config)
    save_name = filename + '.ckpt'

    model = SimCLR_pl(train_config, model=resnet18(pretrained=False), feat_dim=512)

    transform = Augment(train_config.img_size)
    data_loader = get_stl_dataloader(train_config.batch_size, transform)

    accumulator = GradientAccumulationScheduler(scheduling={0: train_config.gradient_accumulation_steps})
    checkpoint_callback = ModelCheckpoint(filename=filename, dirpath=save_model_path,every_n_epochs=2,
                                            save_last=True, save_top_k=2,monitor='Contrastive loss_epoch',mode='min')
   
        
        
        
        

    if resume_from_checkpoint:
        trainer = Trainer(callbacks=[accumulator],# checkpoint_callback],
                      gpus=available_gpus, strategy="ddp", 
                      max_epochs=train_config.epochs,
                      resume_from_checkpoint=train_config.checkpoint_path)
    else:
        trainer = Trainer(callbacks=[accumulator],#, checkpoint_callback],#, checkpoint_callback],
                      gpus=available_gpus, strategy="ddp", 
                      max_epochs=train_config.epochs)
    
     
    import os
    
    trainer._save_checkpoint = trainer.save_checkpoint
    
    def save_checkpoint(
        filepath, weights_only = False, storage_options = None
    ) -> None:
        
        if 'LOCAL_RANK' not in os.environ:
            print("Saving Distributed Checkpoint")
            trainer._save_checkpoint(filepath, weights_only, storage_options)
        elif int(str(os.environ['LOCAL_RANK'])) == 0:
            print("Saving Distributed Checkpoint")
            print(f"WORLD_SIZE {os.environ['WORLD_SIZE']}")
            print(f"LOCAL RANK {os.environ['LOCAL_RANK']}]")
            trainer._save_checkpoint(filepath, weights_only, storage_options)
        else:
            pass
            
    trainer.save_checkpoint = save_checkpoint


    trainer.fit(model, data_loader)


    trainer.save_checkpoint(save_name)