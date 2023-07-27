cd scripts

exp_name=$1
model_base=$2
hidden_vector_size=${3:-64}
num_rnn_layers=${4:-1}
rnn_dropout=${5:-0.2}
num_classification_layers=${6:-2}
max_epochs=${7:-100} # if unset or empty arg set to 100
train_with_val=${8:-False}
inference=${9:-False}

#### Below we always run all different configuration of the datasets to add and on a single-head or multi-head model ###

# Weighted loss function (to adjust for imbalanced datasets)
python models.py --exp_name $exp_name --model_base $model_base --hidden_vector_size $hidden_vector_size --num_rnn_layers $num_rnn_layers --rnn_dropout $rnn_dropout --num_classification_layers $num_classification_layers --max_epochs $max_epochs --train_with_val $train_with_val --inference $inference --weighted_loss_fn --exclude_geowiki
python models.py --exp_name $exp_name --model_base $model_base --hidden_vector_size $hidden_vector_size --num_rnn_layers $num_rnn_layers --rnn_dropout $rnn_dropout --num_classification_layers $num_classification_layers --max_epochs $max_epochs --train_with_val $train_with_val --inference $inference --weighted_loss_fn --geowiki_subset nigeria
python models.py --exp_name $exp_name --model_base $model_base --hidden_vector_size $hidden_vector_size --num_rnn_layers $num_rnn_layers --rnn_dropout $rnn_dropout --num_classification_layers $num_classification_layers --max_epochs $max_epochs --train_with_val $train_with_val --inference $inference --weighted_loss_fn --geowiki_subset neighbours1
python models.py --exp_name $exp_name --model_base $model_base --hidden_vector_size $hidden_vector_size --num_rnn_layers $num_rnn_layers --rnn_dropout $rnn_dropout --num_classification_layers $num_classification_layers --max_epochs $max_epochs --train_with_val $train_with_val --inference $inference --weighted_loss_fn
python models.py --exp_name $exp_name --model_base $model_base --hidden_vector_size $hidden_vector_size --num_rnn_layers $num_rnn_layers --rnn_dropout $rnn_dropout --num_classification_layers $num_classification_layers --max_epochs $max_epochs --train_with_val $train_with_val --inference $inference --weighted_loss_fn --exclude_nigeria --geowiki_subset nigeria
python models.py --exp_name $exp_name --model_base $model_base --hidden_vector_size $hidden_vector_size --num_rnn_layers $num_rnn_layers --rnn_dropout $rnn_dropout --num_classification_layers $num_classification_layers --max_epochs $max_epochs --train_with_val $train_with_val --inference $inference --weighted_loss_fn --exclude_nigeria --geowiki_subset neighbours1
python models.py --exp_name $exp_name --model_base $model_base --hidden_vector_size $hidden_vector_size --num_rnn_layers $num_rnn_layers --rnn_dropout $rnn_dropout --num_classification_layers $num_classification_layers --max_epochs $max_epochs --train_with_val $train_with_val --inference $inference --weighted_loss_fn --exclude_nigeria
python models.py --exp_name $exp_name --model_base $model_base --hidden_vector_size $hidden_vector_size --num_rnn_layers $num_rnn_layers --rnn_dropout $rnn_dropout --num_classification_layers $num_classification_layers --max_epochs $max_epochs --train_with_val $train_with_val --inference $inference --weighted_loss_fn --geowiki_subset neighbours1 --multi_headed
python models.py --exp_name $exp_name --model_base $model_base --hidden_vector_size $hidden_vector_size --num_rnn_layers $num_rnn_layers --rnn_dropout $rnn_dropout --num_classification_layers $num_classification_layers --max_epochs $max_epochs --train_with_val $train_with_val --inference $inference --weighted_loss_fn --exclude_nigeria --multi_headed
python models.py --exp_name $exp_name --model_base $model_base --hidden_vector_size $hidden_vector_size --num_rnn_layers $num_rnn_layers --rnn_dropout $rnn_dropout --num_classification_layers $num_classification_layers --max_epochs $max_epochs --train_with_val $train_with_val --inference $inference --weighted_loss_fn --multi_headed
python models.py --exp_name $exp_name --model_base $model_base --hidden_vector_size $hidden_vector_size --num_rnn_layers $num_rnn_layers --rnn_dropout $rnn_dropout --num_classification_layers $num_classification_layers --max_epochs $max_epochs --train_with_val $train_with_val --inference $inference --weighted_loss_fn --exclude_nigeria --geowiki_subset neighbours1 --multi_headed

# No weighted loss function
python models.py --exp_name $exp_name --model_base $model_base --hidden_vector_size $hidden_vector_size --num_rnn_layers $num_rnn_layers --rnn_dropout $rnn_dropout --num_classification_layers $num_classification_layers --max_epochs $max_epochs --train_with_val $train_with_val --inference $inference --exclude_geowiki
python models.py --exp_name $exp_name --model_base $model_base --hidden_vector_size $hidden_vector_size --num_rnn_layers $num_rnn_layers --rnn_dropout $rnn_dropout --num_classification_layers $num_classification_layers --max_epochs $max_epochs --train_with_val $train_with_val --inference $inference --geowiki_subset nigeria
python models.py --exp_name $exp_name --model_base $model_base --hidden_vector_size $hidden_vector_size --num_rnn_layers $num_rnn_layers --rnn_dropout $rnn_dropout --num_classification_layers $num_classification_layers --max_epochs $max_epochs --train_with_val $train_with_val --inference $inference --geowiki_subset neighbours1
python models.py --exp_name $exp_name --model_base $model_base --hidden_vector_size $hidden_vector_size --num_rnn_layers $num_rnn_layers --rnn_dropout $rnn_dropout --num_classification_layers $num_classification_layers --max_epochs $max_epochs --train_with_val $train_with_val --inference $inference
python models.py --exp_name $exp_name --model_base $model_base --hidden_vector_size $hidden_vector_size --num_rnn_layers $num_rnn_layers --rnn_dropout $rnn_dropout --num_classification_layers $num_classification_layers --max_epochs $max_epochs --train_with_val $train_with_val --inference $inference --exclude_nigeria --geowiki_subset nigeria
python models.py --exp_name $exp_name --model_base $model_base --hidden_vector_size $hidden_vector_size --num_rnn_layers $num_rnn_layers --rnn_dropout $rnn_dropout --num_classification_layers $num_classification_layers --max_epochs $max_epochs --train_with_val $train_with_val --inference $inference --exclude_nigeria --geowiki_subset neighbours1
python models.py --exp_name $exp_name --model_base $model_base --hidden_vector_size $hidden_vector_size --num_rnn_layers $num_rnn_layers --rnn_dropout $rnn_dropout --num_classification_layers $num_classification_layers --max_epochs $max_epochs --train_with_val $train_with_val --inference $inference --exclude_nigeria
python models.py --exp_name $exp_name --model_base $model_base --hidden_vector_size $hidden_vector_size --num_rnn_layers $num_rnn_layers --rnn_dropout $rnn_dropout --num_classification_layers $num_classification_layers --max_epochs $max_epochs --train_with_val $train_with_val --inference $inference --geowiki_subset neighbours1 --multi_headed
python models.py --exp_name $exp_name --model_base $model_base --hidden_vector_size $hidden_vector_size --num_rnn_layers $num_rnn_layers --rnn_dropout $rnn_dropout --num_classification_layers $num_classification_layers --max_epochs $max_epochs --train_with_val $train_with_val --inference $inference --exclude_nigeria --multi_headed
python models.py --exp_name $exp_name --model_base $model_base --hidden_vector_size $hidden_vector_size --num_rnn_layers $num_rnn_layers --rnn_dropout $rnn_dropout --num_classification_layers $num_classification_layers --max_epochs $max_epochs --train_with_val $train_with_val --inference $inference --multi_headed
python models.py --exp_name $exp_name --model_base $model_base --hidden_vector_size $hidden_vector_size --num_rnn_layers $num_rnn_layers --rnn_dropout $rnn_dropout --num_classification_layers $num_classification_layers --max_epochs $max_epochs --train_with_val $train_with_val --inference $inference --exclude_nigeria  --geowiki_subset neighbours1 --multi_headed

# Parse json results and generate csv file
python parse_results.py $exp_name $model_base