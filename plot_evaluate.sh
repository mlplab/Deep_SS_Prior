#!/usr/bin/zsh


CMDNAME=`basename $0`


datasets="CAVE Harvard"
# model_name="FusionReconst"
block_num=9
feature_blocks=(1 2 3 4)
learned_time="0513"
echo $learned_time


model_name=( `echo $model_name | tr ' ' ' '` )
datasets=( `echo $datasets | tr ' ' ' '` )
modes=( `echo $modes | tr ' ' ' '` )
for name in $model_name[@]; do
    echo $name
done
for dataset in $datasets[@]; do
    for name in $model_name[@]; do
        if [ $name = "FusionReconst" ]; then
            for feature_block in $feature_blocks[@]; do
                python plot_evaluate.py -d $dataset -c $concat -m $name -b $block_num --learned_time $learned_time -fb $feature_block
            done
        else
            python plot_evaluate.py -d $dataset -c $concat -m $name -b $block_num -lt $learned_time
        fi
    done
done
