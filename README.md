# SM-HK
Code for our paper:
SM-HK: Sentiment-aware Model for Early Depression Risk Detection

## Abstract
With the booming development of social media, massive amounts of user posting data have brought hope for detection of early depressed users. However, the data relied on for depression detection mainly comes from users who have been suffering from depression for a long time and whose statements are full of significant negative depressive sentiment, making them difficult to use to detect whether a user is in the early depression. Moreover, existing methods have mainly focused on modelling semantic information on post texts while neglecting key information such as sentiment knowledge, depression domain knowledge and negative sentiment severity. In this work, we redefine depression detection as early depression risk detection and construct an associated early depression risk detection dataset. Meanwhile, we propose a sentiment-aware model fusing heterogeneous knowledge for depression risk prediction. The model fully and effectively fuses sentiment knowledge with depression domain knowledge by employing a multi-level knowledge fusion strategy to deeply explore the implicit representation of early depressed users. In addition, we propose an auxiliary task for user negative sentiment severity detection to effectively perceive the user's overall negative sentiment severity. The experimental results show that the model achieves state-of-the-art performance on the proposed dataset. In several experiments, we also verify the validity of the constructed dataset and the interpretability of the model.


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

