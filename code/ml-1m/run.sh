nohup python -u main.py --model=$1 --dataset=$2 --lr=$3 --batch_size=$4 --dropout=$5 --alpha=$6 --lamda=$7 --gpu=$8 > ./log/$1_$2_$3lr_$4bs_$5dropout_$6alpha_$7lamda.txt 2>&1 &
