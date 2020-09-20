python open_lth.py lottery_branch retrain \
	--default_hparams cifar_resnet_20 --lr 0.03 --training_steps 30000it --milestone_steps 20000it,25000it --warmup_steps 20000it --rotate_array 10,40,90 \
	--retrain_t_lr 0.03 --retrain_t_training_steps 30000it --retrain_t_milestone_steps 20000it,25000it --retrain_t_warmup_steps 20000it  --levels 5-10 --retrain_d_dataset_name cifar10 --retrain_d_batch_size 128 --retrain_t_optimizer_name sgd --retrain_t_gamma 0.1 --retrain_t_weight_decay 0.0001
