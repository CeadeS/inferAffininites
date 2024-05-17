#1 Install pytorch
#2 Install https://github.com/CeadeS/ImageZipDataset

# The code is adaptedfrom https://github.com/pytorch/vision/blob/main/references/classification/train.py


## This training needs several machines with multiple GPUs per machine.
## run srv_de.sh on the main training machine and cnt_def.sh on the secondary machines
## For single Machine use NUM_NODES = 1

## run on the main server: srv_def.sh -n NUM_NODES
## run on a client machine: cnt_def.sh -n NUM_NODES -h SRV_HOSTNAME

## NUM_NODES is the number of Machines
## SRV_HOSTNAME is the name of the main server
## Ff you have not 8 GPU per Machine, change --nproc_per_node in the*.sh files to the number of GPU you have




