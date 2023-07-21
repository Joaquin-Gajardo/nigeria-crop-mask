cd scripts

exp_name=$1
model_base=$2

# Weighted loss function
python models.py --exp_name $exp_name --model_base $model_base --weighted_loss_fn --exclude_geowiki
python models.py --exp_name $exp_name --model_base $model_base --weighted_loss_fn --geowiki_subset nigeria
python models.py --exp_name $exp_name --model_base $model_base --weighted_loss_fn --geowiki_subset neighbours1
python models.py --exp_name $exp_name --model_base $model_base --weighted_loss_fn
python models.py --exp_name $exp_name --model_base $model_base --weighted_loss_fn --exclude_nigeria --geowiki_subset nigeria
python models.py --exp_name $exp_name --model_base $model_base --weighted_loss_fn --exclude_nigeria --geowiki_subset neighbours1
python models.py --exp_name $exp_name --model_base $model_base --weighted_loss_fn --exclude_nigeria
python models.py --exp_name $exp_name --model_base $model_base --weighted_loss_fn --geowiki_subset neighbours1 --multi_headed
python models.py --exp_name $exp_name --model_base $model_base --weighted_loss_fn --multi_headed
python models.py --exp_name $exp_name --model_base $model_base --weighted_loss_fn --exclude_nigeria --geowiki_subset neighbours1 --multi_headed
python models.py --exp_name $exp_name --model_base $model_base --weighted_loss_fn --exclude_nigeria --multi_headed

# No weighted loss function
python models.py --exp_name $exp_name --model_base $model_base --exclude_geowiki
python models.py --exp_name $exp_name --model_base $model_base --geowiki_subset nigeria
python models.py --exp_name $exp_name --model_base $model_base --geowiki_subset neighbours1
python models.py --exp_name $exp_name --model_base $model_base
python models.py --exp_name $exp_name --model_base $model_base --exclude_nigeria --geowiki_subset nigeria
python models.py --exp_name $exp_name --model_base $model_base --exclude_nigeria --geowiki_subset neighbours1
python models.py --exp_name $exp_name --model_base $model_base --exclude_nigeria
python models.py --exp_name $exp_name --model_base $model_base  --geowiki_subset neighbours1 --multi_headed
python models.py --exp_name $exp_name --model_base $model_base --multi_headed
python models.py --exp_name $exp_name --model_base $model_base --exclude_nigeria  --geowiki_subset neighbours1 --multi_headed
python models.py --exp_name $exp_name --model_base $model_base --exclude_nigeria --multi_headed

# Parse json results and generate csv file
python parse_results.py $exp_name $model_base