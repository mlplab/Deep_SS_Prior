#!/usr/bin/zsh


CMDNAME=`basename $0`


batch_size=64
epoch=150
datasets="CAVE Harvard"
concat="False"
model_name="HSCNN DeepSSPrior HyperReconNet FusionReconst"
feature_blocks=(1 2 3 4)
block_num=9
start_time=$(date "+%m%d")


while getopts b:e:d:c:m:bn: OPT
do
    echo "$OPTARG"
    case $OPT in
        b) batch_size=$OPTARG ;;
        e) epoch=$OPTARG ;;
        d) dataset=$OPTARG ;;
        c) concat=$OPTARG ;;
        m) model_name=$OPTARG ;;
        bn) block_num=$OPTARG ;;
        *) echo "Usage: $CMDNAME [-b batch size] [-e epoch]" 1>&2
            exit 1;;
    esac
done


echo $batch_size
echo $epoch
echo $dataset
echo $block_num
echo $start_time


model_name=( `echo $model_name | tr ' ' ' '` )
datasets=( `echo $datasets | tr ' ' ' '` )
for dataset in $datasets[@]; do
    for name in $model_name[@]; do
        echo $model_name
        if [ $name = "FusionReconst" ]; then
            for feature_block in $feature_blocks[@]; do
                python train_sh.py -b $batch_size -e $epoch -d $dataset -c $concat -m $name -bn $block_num -st $start_time -fb $feature_block
            done
        else
            python train_sh.py -b $batch_size -e $epoch -d $dataset -c $concat -m $name -bn $block_num -st $start_time
        fi
    done
done
