python train.py -task ext -mode train -bert_data_path ../bert_data -ext_dropout 0.1 -model_path ../models/bert_ext -lr 3e-3 -visible_gpus 0,1,2 -report_every 50 -save_checkpoint_steps 2000 -batch_size 3000 -train_steps 50000 -accum_count 2 -log_file ../logs/ext_bert -use_interval true -warmup_steps 10000 
