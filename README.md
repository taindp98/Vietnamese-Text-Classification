# The implementation of Vietnamese Text Classification

We aims to build a simple attention based deep neural model to classify the vietnamese texts.

## Table of contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Quickstart](#quickstart)
- [Results](#results)

## Installation
```bash
pip install -r requirements.txt
```

## Dataset
We collected over 400 questions that relates to the university consultancy in Vietnam. After considering the features, we assign them into 4 classes.
| label | content                                                                                                                                                                                                                              |
|-------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0     | cho em hỏi em có thể đăng kí 2 ngành được không em đăng kí 2 ngành điện và ngành dầu khí vd trượt ngành điện thì qua dầu khí được không ạ                                                                                           |
| 1     | thầy cho em hỏi nếu mình đã trúng tuyển chương trình đại trà thì có thể đổi sang chương trình chất lượng cao tiếng anh hoặc chương trình tiên tiến được không ạ                                                                     |
| 2     | cho em hỏi chỉ tiêu ngành khoa học máy tính năm nay bao nhiêu và giới thiệu về ngành khoa học máy tính với ạ                                                                                                                         |
| 3     | cho em hỏi ngành kỹ thật điện điện tử thì có cơ hội việc làm ra sao ạ                                                                                                                                                                |

## Quickstart
**Note**: Step by step for training, please refer the notebook [train_test.ipynb](./code/train_test.ipynb)

We build the model following the configurations:
  ```python
  from model import AttentionModel
  
  model = AttentionModel(
  	batch_size=8, 
  	output_size=3, 
  	hidden_size=128, 
  	vocab_size=325, 
  	embedding_length=400          
  )
  ```

The output of built model is:

  ```bash
  AttentionModel(
    (word_embeddings): Embedding(325, 400)
    (lstm): LSTM(400, 128)
    (label): Linear(in_features=128, out_features=3, bias=True)
  )
  
  -------------------------------------------------------------------------------------------------
        Layer (type)                                   Input Shape         Param #     Tr. Param #
  =================================================================================================
         Embedding-1                                      [30, 22]         130,000               0
              LSTM-2     [22, 30, 400], [1, 30, 128], [1, 30, 128]         271,360               0
            Linear-3                                     [30, 128]             387               0
  =================================================================================================
  Total params: 401,747
  Trainable params: 0
  Non-trainable params: 401,747
  ```

## Results
After training the model with 30 epochs, we evaluate the trained model in the validation subset, it results an exciting performance (89% acc) in the field of classification problem.

![](./data/confusion_matrix.png)


