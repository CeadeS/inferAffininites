Install pytorch
Install https://github.com/CeadeS/ImageZipDataset

To run the experiments in parallel:

NUM_NODES is the number of Machines
SRV_HOSTNAME is the name of the main server
Ff you have not 8 GPU per Machine, change --nproc_per_node in the*.sh files to the number of GPU you have

run on the main server: srv_def.sh -n NUM_NODES
run on a client machine: cnt_def.sh -n NUM_NODES -h SRV_HOSTNAME

For single Machine use NUM_NODES = 1


