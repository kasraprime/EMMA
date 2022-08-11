#!/bin/bash
#SBATCH --job-name=MMA
#SBATCH --mem=80000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=108:00:00
#SBATCH --constraint=rtx_8000|rtx_6000|rtx_2080
#SBATCH --output=results/jobs/slurm-%j.out
# Make sure to make the results/jobs directory before running this script.

export TMPDIR=/scratch/${SLURM_JOB_ID}

# Sample command for 1 job: sbatch run_cluster.sh my-experiment no_neg_sampling 1024 42 0.50

# Frequently changing
exp_name=$1 # exp, debug, delete, analysis, baseline, develop
method=$2
data_type=$3
batch_size=$4
candidate_constraint=$5
negative_sampling=$6 # no_neg_sampling , neg_sampling  , random_neg_sampling 
embed_dim=$7 # 100 , 150 , 300, 550 This is the latent dimension
random_seed=$8 # 32 , 64 , 128
task=$9 # gold , RIVR , gauss_noise , dropout_noise , snp_noise , clean_normalized
# eval_mode=$2 # train , train-test , test
# activation=$9 # softplus, relu, tanh, softmax

eval_mode='train-test' # train , train-test , test
# eval_mode='test'
activation='relu'
optimizer='SGD'
lr=0.001

# setup directories
exp_full_name=$exp_name'-'$method'-'$data_type'-'$batch_size'-'$activation'-'$optimizer'-'$lr'-'$candidate_constraint'-'$task'-'$negative_sampling'-'$embed_dim
results_dir='results/'$exp_name'-'$method'-'$data_type'-'$batch_size'-'$activation'-'$optimizer'-'$lr'-'$candidate_constraint'-'$task'-'$negative_sampling'-'$embed_dim'/seed-'$random_seed'/'
# results_dir='results/duplicate/'$exp_name'-'$method'-'$data_type'-'$batch_size'-'$activation'-'$optimizer'-'$lr'-'$candidate_constraint'-'$task'-'$negative_sampling'-'$embed_dim'/seed-'$random_seed'/'
train_result_dir=$results_dir'train/'
valid_result_dir=$results_dir'valid/'
test_result_dir=$results_dir'test/'
all_results='results/all/'

mkdir -p $results_dir
mkdir -p $train_result_dir
mkdir -p $valid_result_dir
mkdir -p $test_result_dir
mkdir -p $all_results

# Mostly constant
# task='gold' # 'gold' or 'RIVR'
pred=0.50 # 0.40, 0.50, 0.60
per_epoch='best' # 'best' or 'all'
epoch=200
track=1 # 0 to turn off wandb and tensorboard tracking
gpu_num=$(nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1 | head -n 1 | xargs) # which one is free?

sim=(snp_noise gauss_noise dropout_noise RIVR clean_normalized)
real=(gold gold_raw gold_cropped gold_no_crop_old)

if [ $negative_sampling = no_neg_sampling ]
then
continue
# batch_size=64 # change batch size to 1 if not using negative sampling and using vanilla triplet loss
elif [ $negative_sampling = neg_sampling ] && [[ " ${real[*]} " =~ " $task " ]]
then
# batch_size=64 # 64 or 32 or 8
continue
else
batch_size=4
fi

echo "copying data to scratch directory ..."
if [[ " ${real[*]} " =~ " $task " ]]
then
echo "copying gold dataset ..."
mkdir -p $TMPDIR/data/gold
org_data=../../../../data/gold/images
rsync -a $org_data $TMPDIR/data/gold
audio_data=../../../../data/gold/speech_16
rsync -a $audio_data $TMPDIR/data/gold
elif [[ " ${sim[*]} " =~ " $task " ]]
then
echo "copying ${task} dataset ..."
mkdir -p $TMPDIR/data/simulation
org_data=../../../../data/simulation/images
rsync -a $org_data $TMPDIR/data/simulation
fi
echo "Done copying data"
scratch_data=$TMPDIR/data/


python -u ML.py --wandb_track $track --experiment_name $exp_name --epochs $epoch --task $task --data_dir $scratch_data \
--random_seed $random_seed --embed_dim $embed_dim --data_type $data_type --method $method --candidate_constraint $candidate_constraint \
--prediction_thresh $pred --results_dir $results_dir  --exp_full_name $exp_full_name \
--learning_rate $lr --eval_mode $eval_mode --per_epoch $per_epoch --batch_size $batch_size --optimizer $optimizer \
--activation $activation --gpu_num $gpu_num --negative_sampling $negative_sampling 2>&1 | tee -a $results_dir'out.log'

