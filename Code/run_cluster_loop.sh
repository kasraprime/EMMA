#!/bin/bash

# Make sure not to save the model state at all.
for seed in 7 24 42
# for seed in 42
do
# for task in gold_no_crop gold_crop gold_no_crop_old RIVR gauss_noise dropout_noise snp_noise clean_normalized
# for task in gold gold_raw gold_cropped
for task in gold
do
# for neg in no_neg_sampling neg_sampling
for neg in neg_sampling
do
for dim in 1024
do
sbatch run_cluster.sh my-contrastive-loss-cosine-text-anchor $neg $dim $seed $task
done
done
done
done