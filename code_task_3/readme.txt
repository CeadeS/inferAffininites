#1 Install pytorch
#2 Install pytorch_lightning
#3 Install pytorch-lightning-bolts
#4 Install lightning-flash
#5 Install https://github.com/CeadeS/ImageZipDataset

## To run the experiment:
## code adapted from https://theaisummer.com/simclr/ 
## This training needs several machines with multiple GPUs per machine.
## run srv_de.sh on the main training machine and cnt_def.sh on the secondary machines
## run./crv_def.sh -n 8
## run./cnt_def.sh -n 8 -h server_hostname
## n is the number of Machines
## h is the name of the main server
## If you have not 8 GPU per Machine, change --nproc_per_node in the*.sh files to the number of GPU you have

## You can load the created model to extract features.
