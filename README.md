### Modeling Semantic Compositionality with Sememe Knowledge

Code and data for ACL2019 paper 
[Modeling Semantic Compositionality with Sememe Knowledge](https://arxiv.org/pdf/1907.04744.pdf).
### Requirements

- Tensorflow >= 1.13.1
- Python3.6

#### Data

This repo contains three types of data. 

- Data for Semantic Compositionality Degree (SCD)

  - `./SC Degree/scd.txt` Human annotated MWEs with their SCD, constituents and corresponding sememe set.

    The format for each instance is as follows:

    ```
    农民                           ==>{constituent_word_1}
    职位 人 农                      ==>{sememe_set_of_constituent_word_1}
    起义                           ==>{constituent_word_2}
    暴动 事情 政                    ==>{sememe_set_of_constituent_word_2}
    农民起义                        ==>{MWE}
    事情 职位 政 暴动 人 农          ==>{sememe_set_of_MWE}
    3.0                           ==>{SCD_of_the_MWE}
    ```

- Core data for our model

  - `./dataset/HowNet_original_new.txt`  Original HowNet data

  - `./dataset/hownet.txt` preprocessed and flattened HowNet data

  - `./dataset/train.bin` Training data. Use `pickle` to load.

  - `./dataset/test.bin` Test data. Use `pickle` to load.

  - `./dataset/dev.bin` Dev data. Use `pickle` to load.

  - `./dataset/all.bin` All data. Use `pickle` to load.

  - `./dataset/sememe_vector.txt` Pretrained 1335 sememe embeddings, original file download [here](https://cloud.tsinghua.edu.cn/d/76ab4a71efa541bd8eb3/).

  - `./dataset/word_embedding.txt.zip` Pretrained 200d GloVe embedding. Unzip it before use.

- Filtered word pairs with human annotated similarity data:
  - `./wordsim/filtered_wordsim240.txt   `
  - `./wordsim/filtered_wordsim240.txt   `
  - `./wordsim/COS960.txt   `

Sememe-based Semantic Compositionality Degree

To compare the correlation between human annotated SCD and our proposed sememe-based SCD, please:

```
cd 'SC Degree'
python test_scd.py
```

### MWE Similarity Computation

We use Wordsim240, Wordsim297 and COS960 to test our models performance on MWE similarity computation task. We remove the words in above three dataset which are not MWEs in our dataset and manually move the MWEs in above three dataset to test set.

To run our four models for training on similarity computation task, you could run the following commands: 

SC-AS: 

```
python ps_SC_AS.py
```

SC-MSA:

```
python ps_SC_MSA.py
```

SC-AS+R

```
python ps_SC_AS_R.py
```

SC-MSA+R

```
python ps_SC_MSA_R.py 
```

To evaluate the learned MWE embeddings, please:

```
python eval_wordsim.py {saved_MWE_embedding_path} 
```

### MWE Sememe Prediction

To train and test our models on MWE sememe prediction task, you could run the following commands:

SC-AS: 

```
python sem_SC_AS.py
```

SC-MSA:

```
python sem_SC_MSA.py
```

SC-AS+R

```
python sem_SC_AS_R.py
```

SC-MSA+R

```
python sem_SC_MSA_R.py 
```

### Cite

If you use the code or data, please cite this paper:

```
@inproceedings{Qi2019ModelingSC,
title={Modeling Semantic Compositionality with Sememe Knowledge},
author={Fanchao Qi and Junjie Huang and Chenghao Yang and Zhiyuan Liu and Xiao Chen and Qun Liu and Sun Maosong},
booktitle={Proceedings of ACL 2019}
year={2019}
}
```

