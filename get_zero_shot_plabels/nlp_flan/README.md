# Zero-shot classification with CLIP on NLP datasets

This directory contains code for generating zero-shot classification pseudo-labels with CLIP model on on AG news, Yahoo Answers, and Amazon Review datasets.

## Prepare Dataset
Please follow dataset setup instructions in [train_zeromatch](../../train_zeromatch).

## Conda / pip environment setup
After activating your Python/Conda environment, install the required packages by running:
```shell
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Running inference
To request inference, execute the following commands (with your Python/Conda environment activated):

```
bash run_inference.sh
```

Please take a look at each script files to check modifiable experiment parameters (e.g. selecting dataset, selecting train set or test set, etc.) and change them for your needs.

