#!/bin/bash
#SBATCH --job-name=sim2real-gold
#SBATCH --mem=80000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=108:00:00
#SBATCH --constraint=rtx_8000|rtx_6000|rtx_2080
#SBATCH --output=results/jobs/slurm-%j.out
# Make sure to make the results/jobs directory before running this script.

export TMPDIR=/scratch/${SLURM_JOB_ID}

# Sample command for 1 job: sbatch run_cluster.sh my-experiment no_neg_sampling 1024 42 0.50

exp_name=$1 # softmax, baseline, frozen, develop
negative_sampling=$2 # no_neg_sampling , neg_sampling  , random_neg_sampling 
embed_dim=$3 # 100 , 150 , 300, 550 This is the latent dimension
random_seed=$4 # 32 , 64 , 128
task=$5 # gold , RIVR , gauss_noise , dropout_noise , snp_noise
# eval_mode=$2 # train , train-test , test
# activation=$9 # softplus, relu, tanh

eval_mode='train-test' # train , train-test , test
activation='relu'

# setup directories
exp_full_name=$exp_name'-'$task'-'$negative_sampling'-'$embed_dim
results_dir='results/'$exp_name'-'$task'-'$negative_sampling'-'$embed_dim'/seed-'$random_seed'/'
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
lr=0.001
track=1 # 0 to turn off wandb and tensorboard tracking
gpu_num=$(nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1 | head -n 1 | xargs) # which one is free?

# Frequently changing
data_type='rgbd' # 'rgb', 'depth', 'rgbd'
per_epoch='best' # 'best' or 'all'
epoch=200
if [ $negative_sampling = no_neg_sampling ]
then
batch_size=1 # batch size has to be 1 if not using negative sampling
elif [ $negative_sampling = neg_sampling ] && [ $task = gold ]
then
batch_size=64 # 64 or 32 or 8
else
batch_size=4
fi

sim=(snp_noise gauss_noise dropout_noise RIVR)
echo "copying data to scratch directory ..."
if [ $task = gold ]
then
echo "copying gold dataset ..."
mkdir -p $TMPDIR/data/gold
org_data=../../../../data/gold/images
rsync -a $org_data $TMPDIR/data/gold
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
--random_seed $random_seed --embed_dim $embed_dim --data_type $data_type \
--prediction_thresh $pred  --results_dir $results_dir  --exp_full_name $exp_full_name \
--learning_rate $lr --eval_mode $eval_mode  --per_epoch $per_epoch --batch_size $batch_size \
--activation $activation --gpu_num $gpu_num --negative_sampling $negative_sampling 2>&1 | tee -a $results_dir'out.log'

