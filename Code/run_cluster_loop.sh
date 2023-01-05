#!/bin/bash

# Make sure not to save the model state at all.
for seed in 7 24 42 3407 123
# for seed in 7 24 3407 123
# for seed in 42
do
# for task in gold_no_crop gold_crop gold_no_crop_old RIVR gauss_noise dropout_noise snp_noise clean_normalized
# for task in gold gold_raw gold_cropped
for task in gold
do
# for neg in no_neg_sampling neg_sampling
# for neg in neg_sampling
for neg in no_neg_sampling
do
for dim in 1024
do
# for method in full-emma bce-emma supcon-emma supcon extended-triplet eMMA-text-anchor contrastive-text-anchor supcon-emma-pull-neg
# for method in full-emma-pull-neg
# for method in supcon-emma-pull-neg
# for method in supcon-emma
for method in supcon-emma supcon
# for method in contrastive-org
do
# for modalities in lard lrd ard lar lad lr ld ar ad
for modalities in lard
# for modalities in lrd ard lar lad lr ld ar ad
do
# for batch_size in 64 32 16 8 4 2
for batch_size in 64
do
# for candidate_constraint in unique_objects unique_instances
for candidate_constraint in unique_objects
do
sbatch run_cluster.sh exp-noisy-txt-varied $method $modalities $batch_size $candidate_constraint $neg $dim $seed $task
done
done
done
done
done
done
done
done