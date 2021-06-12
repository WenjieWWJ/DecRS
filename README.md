# Deconfounded Recommendation for Alleviating Bias Amplification



This is the pytorch implementation of our paper at KDD 2021:

> [Deconfounded Recommendation for Alleviating Bias Amplification](https://arxiv.org/abs/2105.10648)
>
> Wenjie Wang, Fuli Feng, Xiangnan He, Xiang Wang, Tat-Seng Chua.

## Environment

- Anaconda 3
- python 3.7.3
- pytorch 1.4.0
- numpy 1.16.4

## Usage

### Data

The experimental data are in './data' folder, except 'item_feature_file.npy' of amazon_book. It is uploaded to [Google drive:DecRS/data/amazon_book](https://drive.google.com/drive/folders/1xww1VA-6Rc911vsAhucA7hi-2Np9zyWX?usp=sharing) due to the large size. 

### Training

```
python main.py --model=$1 --dataset=$2 --lr=$3 --batch_size=$4 --dropout=$5 --alpha=$6 --lamda=$7 --gpu=$8
```

or use run.sh

```
sh run.sh model dataset lr batch_size dropout alpha lamda gpu_id
```

- The log file will be in the './code/{dataset}/log/' folder. 
- The explanation of hyper-parameters can be found in './code/{dataset}/main.py'. 
- The default hyper-parameter settings are detailed in './code/{dataset}/hyper-parameters.txt'.

### Inference

1. Download the ranking scores released by us from [Google drive:DecRS/ranking_scores/{dataset}/](https://drive.google.com/drive/folders/1xww1VA-6Rc911vsAhucA7hi-2Np9zyWX?usp=sharing).
2. Put **four** '.npy' file into the corresponding folder in './code/{dataset}/inference'.
3. Get the results of DecRS over different user groups by runing DecFM.py or DecNFM.py:

```
python DecFM.py 
```

### Examples

1. Train DecFM on ML-1M:

```
cd ./code/ml-1m
sh run.sh DecFM ml_1m 0.05 1024 [0.3,0.3] 0.2 0.1 0
```

2. Inference DecNFM on amazon_book

```
cd ./code/amazon-book/inference
python DecNFM.py
```

## Citation  

If you use our code or data, please kindly cite:

```
@inproceedings{wang2021deconfounding,
  title={Deconfounded Recommendation for Alleviating Bias Amplification},
  author={Wenjie Wang, Fuli Feng, Xiangnan He, Xiang Wang, and Tat-Seng Chua},
  booktitle={KDD},
  year={2021},
  publisher={ACM}
}
```

## Acknowledgment

Thanks to the FM/NFM implementation:

- [NFM-torch](https://github.com/guoyang9/NFM-pyorch/) from Yangyang Guo.
- [NFM-tensorflow](https://github.com/hexiangnan/neural_factorization_machine) from Xiangnan He. 

## License

NUS © [NExT++](https://www.nextcenter.org/)