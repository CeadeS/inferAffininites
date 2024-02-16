#!/bin/bash
#conda activate ml_poldiv_py310   &&
#cd /home/mhofmann/nas/home/py_projects/dist_test &&

set -u

print_usage(){
	    echo "usage: script [-n nnodes]"
    }
if [[ $# -eq 0 ]]; then
	    print_usage
	        exit 1
fi

nnodes=2
host_name="localhost"

while getopts n: flag 
do
	case $flag in
	     n) nnodes=$OPTARG;;
	esac
done
echo "$nnodes";

env NCCL_SOCKET_IFNAME=bond0 NCCL_DEBUG="INFO" torchrun --nnodes $nnodes --nproc_per_node 8 --node_rank 0 --max_restarts 1 --rdzv_id 1 --rdzv_backend c10d --rdzv_endpoint localhost:20010 pre_train.py