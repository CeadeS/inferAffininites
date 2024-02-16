#!/bin/bash
#conda activate ml_poldiv_py310   &&
#cd /home/mhofmann/nas/home/py_projects/dist_test &&

set -u

print_usage(){
	    echo "usage: script [-n nnodes] [-h hostname]"
    }
if [[ $# -eq 0 ]]; then
	    print_usage
	        exit 1
fi

nnodes=2
host_name="betty5"

while getopts h:n: flag 
do
	case $flag in
	     h) host_name=$OPTARG;;
	     n) nnodes=$OPTARG;;
	esac
done
env NCCL_SOCKET_IFNAME=bond0 NCCL_DEBUG="INFO" torchrun --nnodes $nnodes --nproc_per_node 8 --node_rank 1 --max_restarts 1 --rdzv_id 1 --rdzv_backend c10d --rdzv_endpoint "$host_name":20010 pre_train.py
