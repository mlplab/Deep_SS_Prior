batch_size=64
epoch=150
datasets="CAVE Harvard"
concat="False"
model_name="HSCNN DeepSSPrior HyperReconNet FusionReconst"
feature_blocks=(1 2 3 4)
block_num=9
start_time=$(date "+%m%d")


for dataset in $datasets; do
    echo $dataset
    ckpt_path="../SCI_ckpt/${dataset}_${start_time}"
    all_trained_path="${ckpt_path}/all_trained"
    for name in $model_name; do
        echo $model_name
        if [ ! -e "${all_trained_path}/${model_name}_09.tar" ]; then
            echo "${all_trained_path}/${model_name}_09.tar is not exist"
        else
            echo "${all_trained_path}/${model_name}_09.tar is exist"
        fi
    done
done
