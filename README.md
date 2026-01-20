First Change the data path directories for training and validation datsets 
in args: --traindir and --valdir

1. RSCL pretraining + prototypical fine-tuning (default args from args.py):
python3 train.py --model_name="ckpt"

Evaluation -
a. protoloss: (scl method)

eval with no freezing:
python3 eval_finetune.py --model_name="ckpt" --csv_file="protoLoss_no_frozen" 

eval with FREEZING:
python3 eval_finetune.py --ft=0 --model_name="ckpt" --csv_file="protoLoss_frozen"

b. CE with linear classifier:

eval with no freezing:
python3 eval_finetune.py --ftmethod="ce" --model_name="ckpt" --csv_file="ceLoss_no_frozen" 

eval with FREEZING:
python3 eval_finetune.py --ftmethod="ce" --ft=0 --model_name="ckpt" --csv_file="ceLoss_frozen"


2. Three-view RSCL pretraining with an additional augmentation transform: (BEST MODEL)
python3 train.py --epochs=150 bs=64 --nTrainingViews=3 --nTransforms=3 --model_name="3rdtransform_pretraining"

eval with no freezing:
python3 eval_finetune.py --model_name="3rdtransform_pretraining" --csv_file="eval_3rdtransform_pretraining" 


3. Four-view RSCL pretraining using same transorms (same augemntation pipelines used in (1)):
python3 train.py --epochs=150 bs=64 --nTrainingViews=4 --model_name="4views_pretraining"

eval with no freezing:
python3 eval_finetune.py --model_name="4views_pretraining" --csv_file="eval_4views_pretraining"

model_name and csv_file name are just for reference of how they have been saved by us. you can change them as your preference accordingly.






