python open_lth.py lottery \
	        --default_hparams cifar_resnet_20 --batch_size 512 --lr 0.1 --training_steps 7500it --milestone_steps 5000it,6250it --warmup_steps 5000it --levels 20 --num_workers 16 --blur_factor 5 \ 
		--retrain_t_lr 0.1 --retrain_t_training_steps 7500it --retrain_t_milestone_steps 5000it,6250it --retrain_t_warmup_steps 5000it  --levels 0-20 --retrain_d_dataset_name cifar10 --retrain_d_batch_size 512 --retrain_t_optimizer_name sgd --retrain_t_gamma 0.1 --retrain_t_weight_decay 0.0001
