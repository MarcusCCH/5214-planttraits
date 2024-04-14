
# 1000 samples
python3 main.py --train_csv data/train_1000.csv --batch_size 2 --device cuda:1

# 10000 samples 
python3 main.py --train_csv data/train_10000.csv --batch_size 16 --device cuda:1

# 30000 samples
python3 main.py --train_csv data/train_30000.csv --batch_size 32 --device cuda:1 --case_name train_30000 --pretrain output/train_30000/model_0.pth

# 50000 samples
python3 main.py --train_csv data/train_50000.csv --batch_size 32 --device cuda:1 --case_name train_50000 --pretrain output/train_30000/model_4.pth

# full training
# python3 main.py --train_csv data/train.csv --batch_size 32 --device cuda:1 --case_name train_full --pretrain output/train_full/model_56.pth
# python3 main.py --train_csv data/train.csv --batch_size 64  --case_name vit_b32  --device cuda:2
# python3 main.py --train_csv data/train.csv --batch_size 64  --img 224 --case_name vit_b_16  --device cuda:2 --pretrain output/vit_b_16/model_5.pth
# python3 main.py --train_csv data/train.csv --batch_size 64  --img 224 --case_name vit_b_16 --model vit_b_16 --device cuda:2 

python3 main.py --train_csv data/train.csv --batch_size 64  --case_name eff_s_326 --model eff --device cuda:2
python3 main.py --train_csv data/train.csv --batch_size 64  --case_name eff_s_326_out98 --model eff --device cuda:2 --pretrain output/eff_s_326_out98/model_0.pth
python3 main.py --train_csv data/train.csv --batch_size 64  --case_name eff_s_512_out98 --model eff --hidden 512 --device cuda:3 --pretrain output/eff_s_512_out98/model_0.pth
python3 main.py --train_csv data/train.csv --batch_size 32  --case_name eff_l_512_out98 --model eff_l --hidden 512 --device cuda:3 --pretrain output/eff_l_512_out98/model_0.pth

# evaluation
# python3 evaluation.py --model vit --model_dir output/vit_b32/  --device cuda:3
# python3 evaluation.py --model vit_b_16 --model_dir output/vit_b_16/  --device cuda:3
# python3 evaluation.py --model vit_b_16 --model_dir output/vit_b_16/ --model_path output/vit_b_16/model_7.pth --device cuda:3
python3 evaluation.py --model eff --model_dir output/eff_s_326/  --device cuda:3
python3 evaluation.py --model eff --model_dir output/eff_s_326/  --model_path output/eff_s_326_out98/model_90.pth --device cuda:3
python3 evaluation.py --model eff --model_dir output/eff_s_326_out98  --device cuda:2
python3 evaluation.py --model eff --model_dir output/eff_s_512_out98  --hidden 512 --device cuda:2
