# ZeroMatch Training

This directory includes code to reproduce the training results of ZeroMatch.
The codebase is built on top of the [Unified Semi-supervised Learning codebase](https://github.com/microsoft/Semi-supervised-learning). Additionally, we include the Flowers102 and Resisc45 datasets from the FRAMED benchmark introduced in GRIP ([paper](https://papers.neurips.cc/paper_files/paper/2023/file/bf85879363044ca21f7868a3d1b4021c-Paper-Conference.pdf), [code](https://github.com/BatsResearch/menghini-neurips23-code)).

## Prepare Dataset
Please download the dataset tar files:
- [data_zeromatch_cifar.tar](https://drive.google.com/file/d/1cyFGtCIzp1Yk-IM3rmposNNhpp33XM3Z/view?usp=sharing): includes the CIFAR100 dataset from USB.
- [data_zeromatch_framed.tar](https://drive.google.com/file/d/1e5IpLBT2n98dKM5go2Cuixyi1QyLaWED/view?usp=sharing): includes the Flowers102 and Resisc45 datasets from the [GRIP codebase](https://github.com/BatsResearch/menghini-neurips23-code).
- [data_zeromatch_nlp.tar](https://drive.google.com/file/d/1rC4qusGOfzhHeNQKpcvKQ7yYbHMxsZwC/view?usp=sharing): includes the Yahoo Answers, AG News, and Amazon Review datasets from USB.

If the above datasets are not available, you can follow the data setup instructions from [USB](https://github.com/microsoft/Semi-supervised-learning) and [GRIP](https://github.com/BatsResearch/menghini-neurips23-code).  
Each tar file includes zero-shot pseudo-label sets for the corresponding datasets in the `plabels` directory.

After downloading the tar files, extract them to the `../scratch` directory, so that the directory tree of `../scratch` looks as follows:
```
$ tree -L 2 ../scratch
../scratch
├── data_zeromatch_cifar
│   ├── cifar100
│   └── plabels
├── data_zeromatch_framed
│   ├── Flowers102
│   ├── RESICS45
│   ├── cifar100
│   ├── class_files
│   └── plabels
└── data_zeromatch_nlp
    ├── ag_news
    ├── amazon_review
    ├── plabels
    └── yahoo_answers
```


## Conda / pip environment setup
After activating your Python/Conda environment, install the required packages by running:
```shell
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Running training script
To run ZeroMatch training, execute the following commands (with your Python/Conda environment activated):

```shell
# for training CIFAR100
bash scripts_zm/train_cifar.sh 

# for training flowers102, resisc45
bash scripts_zm/train_framed.sh 

# for training Yahoo Answers, AG News, Amazon Review
bash scripts_zm/train_nlp.sh 
```

Please take a look at each script file to check modifiable experiment parameters (e.g., selecting datasets, increasing label size, selecting foundation models, etc.) and adjust them as needed.

Note that each script runs a short training test by default. To run the full training experiment, remove the following argument line in each command: (`--epoch_ow 4 --num_train_iter_ow 96 --num_eval_iter_ow 32 --num_log_iter_ow 8`).

