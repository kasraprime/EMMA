#!/bin/bash

# Make sure not to save the model state at all.
for seed in 7 24 42 3407 123
# for seed in 7
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
# for method in full-emma bce-emma supcon-emma supcon extended-triplet eMMA-text-anchor contrastive-text-anchor
for method in full-emma-pull-neg
do
# for modalities in lard lrd ard lar lad lr ld ar ad
for modalities in lard
do
# for batch_size in 64 32 16 8 4 2
for batch_size in 64
do
# for candidate_constraint in unique_objects unique_instances
for candidate_constraint in unique_objects
do
sbatch run_cluster.sh exp $method $modalities $batch_size $candidate_constraint $neg $dim $seed $task
## sbatch run_cluster.sh 3M-contrastive-SGD-cosine-submodalities-text-anchor $neg $dim $seed $task
# sbatch run_cluster.sh 4M-contrastive-cosine-submodalities-text-anchor-similar-objects $neg $dim $seed $task
# sbatch run_cluster.sh 4M-contrastive-cosine-submodalities-text-anchor-unique-object $neg $dim $seed $task
# sbatch run_cluster.sh 4M-eMMA-cosine-submodalities-text-anchor-similar-objects $neg $dim $seed $task
# sbatch run_cluster.sh 4M-full-eMMA-SpeechRGB-cosine-distance-pull-neg-BS64-submodalities-unique-object $neg $dim $seed $task $method $modalities $batch_size
# sbatch run_cluster.sh 4M-extended-triplet-eMMA-cosine-distance-submodalities-unique-object $neg $dim $seed $task
# sbatch run_cluster.sh 4M-supcon-SGD-normalize-BS2-submodalities-text-first-unique-objects $neg $dim $seed $task
# sbatch run_cluster.sh 4M-eMMA-BCE-temperature0.07-cosine-distance-submodalities-unique-object $neg $dim $seed $task
# sbatch run_cluster.sh 4M-supcon-SGD-cosine-submodalities-text-first-unique-object $neg $dim $seed $task
done
done
done
done
done
done
done
done