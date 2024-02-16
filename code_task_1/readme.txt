Steps
#1 Install python~3.10 and Pytorch
#2 Install libjpegturbo libjpegtran
#3 Install pytorch-h5dataset and tar-dir-indexer with pip install pytorch-h5dataset tar-dir-indexer
#4 Acquire the Data from Steffen Kiel in h5 format. 
#5 extract config.zip into the root folder
#5 run python python train.py -m exp=fafos

## Config Files
## the exp=<EXPNAME> is the file name of one of the configuration files in the config/exp folder
## Names of the config files refer to the experiment settings
## fafos means family zero shot trained on family order and subclass
## naa means naive (80/20) split, all taxon levels traines, all taxon levels evaluated
