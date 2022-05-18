export PYTHONPATH=/home/$USER/MCL:$PYTHONPATH
# domainnet
python -u train_mcl.py \
		--dataset multi \
		--base_path ./data/txt/multi/ \
		--data_root your_data_path \
		--source clipart \
		--target sketch \
		--num 3 \
		--log_dir ./logs \
		--lambda_cls 1 \
		--T2 1 \
		--seed 0 \

# officehome
python -u train_mcl.py \
		--dataset office_home \
		--base_path ./data/txt/office_home/ \
		--data_root your_data_path \
		--source Clipart \
		--target Art \
		--num 3 \
		--log_dir ./logs \
		--lambda_cls 0.2 \
		--T2 1.25 \
		--seed 0 \