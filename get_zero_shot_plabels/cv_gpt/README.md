# Zero-shot classification with GPT-4.1 on CIFAR and FRAMED.

This directory contains code for generating zero-shot classification pseudo-labels with GPT-4.1 model on CIFAR100, Flowers102, and Resisc45 dataset.

## Prepare Dataset
Please follow dataset setup instructions in [train_zeromatch](../../train_zeromatch).

## Conda / pip environment setup
After activating your Python/Conda environment, install the required packages by running:
```shell
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Prepare Model inference server API access.
The inference code uses OpenAI client to run LLM inference in inference API servers. For GPT model, you'll need API keys to access GPT API. For Llama model, you'll need API base url and API keys to run inference (with inference services such as [Microsoft Azure AI](https://learn.microsoft.com/en-us/rest/api/aifoundry/modelinference/) and others)

Set the API base url and API keys to run_inference.sh as following:

```shell
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="your api key here"
```

## Running inference
To request inference, execute the following commands (with your Python/Conda environment activated):

```
bash run_inference.sh
```

Please take a look at each script files to check modifiable experiment parameters (e.g. selecting dataset, selecting train set or test set, etc.) and change them for your needs.




