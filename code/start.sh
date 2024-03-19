#!/usr/bin/env bash

#SBATCH --job-name=classification
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:0
#SBATCH -o ./log_%j.out # STDOUT out
#SBATCH -e ./log_%j.err # STDERR out
#SBATCH --account=COMS030144
#SBATCH --mem=20gb
 
 
echo start time is "$(date)"
echo Slurm job ID is "${SLURM_JOBID}"
 

module add lang/python/anaconda/pytorch
module add lang/cuda/11.2.2
source activate pytorch
export PYTHONPATH=/user/home/aw21131/.conda/envs/pytorch/lib/python3.7/site-packages:$PYTHONPATH
#heatmap_decoder_mse seed1 no freeze 100 epochs
python -u main_landmarks_X3D_losscurve.py --exp_name baseline_focal_decoder_80 --seed 1 --embedding decoder --class_idx 0 --input_sample 3 --epochs 100 --batch_size 1 --lr 0.001 --use_landmark True 

#heatmap_decoder_mse seed1 only retraining regression head 30 epochs 1e-4 (waiting for the upper weights)
# python -u main_landmarks_X3D_losscurve.py --exp_name baseline_gai_SA --seed 1 --embedding SA --class_idx 0 --input_sample 3 --batch_size 4 --lr 0.001 --bmse

# Pain only baseline
# python -u pain_landmarks_X3D_losscurve.py --exp_name newaug23_80_baseline --clip_len 80 --seed 1 --score VAS --fold 3 --embedding none --input_sample 2 --batch_size 4 --lr 0.001 
# python -u pain_landmarks_megaPretrainModel.py --exp_name r3d18_21_pain_3fc20_upl4 --clip_len 144 --seed 1 --score VAS --fold 1 --embedding none --input_sample 2 --batch_size 4 --lr 0.001 
# python -u pain_landmarks_uniformer.py --exp_name uniformer_20 --clip_len 144 --seed 1 --score VAS --fold 0 --embedding none --input_sample 2 --batch_size 1 --lr 4e-4 --wd 0.05 

# python -u pain_landmarks_X3D_losscurve.py --exp_name baseline_SA_ib_144_20 --seed 1 --score VAS --fold 1 --clip_len 144 --embedding decoder --input_sample 2 --batch_size 4 --lr 0.001 --use_landmark True
# python -u pain_landmarks_X3D_losscurve.py --exp_name baseline_23_decoder_gai --fold 3 --input_sample 2 --seed 1 --clip_len 144 --embedding decoder --score VAS --batch_size 4 --lr 0.001 --use_landmark True


# python main_landmarks_LSTM.py --exp_name Former_dual_coor_sum_60 --embedding sum --class_idx 4 --input_sample 3 --b 4 --lr 0.001 --use_landmark True 
# python main_landmarks_LSTM.py --exp_name Former_dual_coor_cat_3loss --embedding concat --class_idx 4 --input_sample 3 --b 4 --lr 0.001 --use_landmark True
# python main_landmarks_LSTM.py --exp_name onlylandmark_LSTM --embedding none --class_idx 4 --input_sample 3 --b 4 --lr 0.001 --use_landmark True --landmark_only True 
# python main_landmarks_X3D.py --exp_name heatmap_only_noweights_nocentercrop --embedding none --class_idx 4 --input_sample 3 --b 4 --lr 0.001 --use_landmark True --landmark_only True
# python main_landmarks_X3D_losscurve.py --exp_name 101slowonly_heatmap_sum_2loss_curve --embedding sum --class_idx 4 --input_sample 3 --b 4 --lr 0.001 --use_landmark True 
# python main_landmarks_X3D_losscurve.py --exp_name heatmap_decoder_noweights --embedding decoder --class_idx 0 --input_sample 3 --b 4 --lr 0.001 --use_landmark True
# python -u main_landmarks_X3D_losscurve.py --exp_name base_heatmap_decoder_gai_noise1_fine --seed 407 --embedding decoder --class_idx 0 --input_sample 3 --batch_size 4 --lr 0.001 --use_landmark True --bmse --noise_sigma 1.

#cd python file path
#cd ./scratch
 
#python -u train.py --net Softsplat --data_dir /mnt/storage/home/mt20523/scratch --out_dir #train_results/softsplat/ --epochs 85 --loss 1*Lap --lr 0.000125 --lr_decay 10 --batch_size 8 --#optimizer ADAMax --load train_results/softsplat/checkpoint/model_epoch075.pth --finetune_pwc


echo end time is "$(date)"
hostname

