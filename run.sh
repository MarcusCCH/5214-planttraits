# 1000 samples
python3 main.py --train_csv train_1000.csv --batch_size 8 --device cuda:1

# 10000 samples 
python3 main.py --train_csv train_10000.csv --batch_size 16 --device cuda:1


# 30000 samples
python3 main.py --train_csv train_30000.csv --batch_size 16 --device cuda:1