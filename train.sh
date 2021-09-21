jobname=resnet
num_gpus=1

export PYTHONPATH=/home/$USER/MCL:$PYTHONPATH

srun -p p-A100 --nodelist=pgpu14 --job-name=$jobname --gres=gpu:$num_gpus -n 1 --ntasks-per-node=1 --cpus-per-task=6 \
python -u train_fix_con_balance.py \
					--source clipart \
					--target sketch \
					--num 3 \
					--log_dir logs3/domainnet/fixmatch/C_S_fix1_mccnorm1_ot0_sampling \
					--lambda_u 1 \
					--lambda_mcc 1 \
					--lambda_entmax 0 \
					--lambda_scatter 0 \
					--lambda_align 0 \
					--lambda_mme 0 \
					--eta_mme 0 \
					--lambda_ot 0 \
					--lambda_g 0 \
					--prop 0.1 \
					--warm 250 \
					--n_workers 4 \
					--seed 0 \


