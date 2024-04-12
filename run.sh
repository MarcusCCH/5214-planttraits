# 1000 samples
python3 main.py --train_csv train_1000.csv --batch_size 2 --device cuda:1

# 10000 samples 
python3 main.py --train_csv train_10000.csv --batch_size 16 --device cuda:1


# 30000 samples
python3 main.py --train_csv train_30000.csv --batch_size 32 --device cuda:1 --case_name train_30000 --pretrain output/train_30000/model_0.pth
