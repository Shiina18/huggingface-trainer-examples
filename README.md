# huggingface-trainer-examples

Huggingface Trainer can be used for customized structures. Read [Huggingface Transformers Trainer as a general PyTorch trainer](https://shiina18.github.io/machine%20learning/2023/01/04/trainer/) for more detail.

The code is organized around huggingface transformers Trainer. Thus, it is modularized, clean, and easy to modify. And the user can enjoy the great logging utility and easy distributed training on multiple GPUs provided by Trainer.

The major dependencies are huggingface transformers and torch. While some bert4torch code is imported, those pieces are in fact short and standalone, and can be copied and pasted with ease if you don't want an extra package.

Why not use bert4torch directly? More standard huggingface transformers integration and clean code are the pursuit of this repository. Yet quite many useful modules and tricks are implemented in bert4torch, so it is still a good reference.
 
## Directory structure

- examples: python scripts
- data: datasets
- pretrained_models: huggingface models

## Datasets

Datasets are majorly available [here](https://github.com/Tongjilibo/bert4torch/blob/master/examples/README.md#二数据集) or as follows.

| Datasets        | Usage     | Downloads                                                                                                                       |
|-----------------|-----------|---------------------------------------------------------------------------------------------------------------------------------|
| 人民日报数据集         | 实体识别      | [china-people-daily-ner-corpus](http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz)                               
| 百度关系抽取          | 关系抽取      | [BD_Knowledge_Extraction](http://ai.baidu.com/broad/download?dataset=sked)                                                      
| Sentiment       | 情感分类      | [Sentiment](https://github.com/bojone/bert4keras/blob/master/examples/datasets/sentiment.zip)                                   
| THUCNews        | 文本分类、文本生成 | [THUCNews](http://thuctc.thunlp.org/#%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE%E9%9B%86THUCNews) 
| ATEC            | 文本相似度     | [ATEC](https://github.com/IceFlameWorm/NLP_Datasets/tree/master/ATEC)                                                           
| BQ              | 文本相似度     | [BQ](http://icrc.hitsz.edu.cn/info/1037/1162.htm)                                                                               
| LCQMC           | 文本相似度     | [LCQMC](http://icrc.hitsz.edu.cn/Article/show/171.html)                                                                         
| PAWSX           | 文本相似度     | [PAWSX](https://arxiv.org/abs/1908.11828)                                                                                       
| STS-B           | 文本相似度     | [STS-B](https://github.com/pluto-junzeng/CNSD)                                                                                  
| CSL             | 文本生成      | [CSL](https://github.com/CLUEbenchmark/CLGE)                                                                                    |
| THUCNews_sample | 文本分类      | [Bert-Chinese-Text-Classification-Pytorch](https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch#中文数据集)         
