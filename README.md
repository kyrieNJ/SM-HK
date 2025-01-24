# SM-HK
Code for our paper:
SM-HK: Sentiment-aware Model Fusing Heterogeneous Knowledge for Depression Risk Detection on Social Media


## Requirements
* Python 3.8
* torch 2.3.0
* SpaCy 3.7.2
* numpy 1.24.3
* argparse 1.4.0
* scikit-learn 1.3.2
* transformers 4.34.1
* zh-core-web-sm 3.7.0
* jieba 0.42.1

## Data pre-processing stage

* Download BERT-Chinese model with this [link](https://huggingface.co/google-bert/bert-base-chinese).
* Download Senticnet_zh with this [link](https://sentic.net/downloads/) and unzip `senticnet_zh.zip`.
* Generate post-symptomatic heterogeneous graphs of data corresponding to the dataset, run the code [DKHG_graph.py](./SM-HK/DKHG_graph.py).

## Sentiment-enhanced domain fine-tuning
*  First generate this domain corpus with negative sentiment knowledge (DC-NSK), run the code [processdata.py](./SM-HK/finetune/processdata.py). Then fine-tune the model, run code [fine-tune.py](./SM-HK/finetune/fine-tune.py).

## Train stage
* You can train the model, run the code [train.py](./SM-HK/train.py).
```bash
python ./SM-HK/train.py 
```
* You can also use the trained model to infer the results for the given data, run the code [infer.py](./SM-HK/infer.py).
```bash
python ./SM-HK/infer.py 
```

## Citation

If our work has been helpful to you, please mark references to our work in your research and thank you for your support.

