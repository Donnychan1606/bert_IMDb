# bert_IMDb

## 使用BERT模型对IMDb评论数据进行二分类（情感极性分析）
环境：Google公司的Colaboratory笔记本，使用GPU进行加速

数据集：train.csv, test.csv 以及 train.pickle使用pickle格式存储

已封装代码：bert_review_classification.ipynb文件

运行结果见上述代码输出结果（设置epoch为3）

模型准确度为88.93%

---

### 仓库目录向导：

1. 根目录下 bert_review_classification.ipynb 文件为完整数据集下，调整超参数至性能最佳的结果

   > 超参数设置如下：
   >
   > * epoch 为3
   > * learning rate 为2e-5
   > * max_seq_length 为128

2. bert_epoch_test 文件夹下为不同epoch的测试结果

3. bert_learning_rate_test 文件夹下为不同learning rate的测试结果

4. bert_max_seq_len_test 文件夹下为不同max_sequence_length的测试结果

5. bert_dataset_size_test 文件夹下为不同数据集规模下的测试结果

6. train.csv 和 test.csv 分别是清洗好的训练集数据和测试集数据

7. dataset_backup 是数据集的其他存储方式数据

8. MachineLearning 文件夹下为机器学习方法

   > 分别是如下算法：
   >
   > * Naive Bayes 朴素贝叶斯
   > * LinearSVC 

9. LSTM 文件夹下使用 LSTM 模型的测试结果
10. CNN   文件夹下使用 CNN  模型的测试结果
11. IMDB_Loader 文件是IMDb数据集的数据预处理文件

---

与其他深度学习方法的对比结果表格如下：

| 模型 | 准确率（accuracy） | loss | f1 score |
| :--: | :----------------: | :--: | :------: |
| Bert |       88.93%       |      |          |
| LSTM |                    |      |          |
| CNN  |                    |      |          |



## 调整超参数测试

#### 1. 调整epoch

| epoch | 准确率（accuracy） | loss | f1 score |
| :---: | :----------------: | :--: | :------: |
|   1   |                    |      |          |
|   2   |                    |      |          |
|   3   |       88.93%       |      |          |
|   6   |                    |      |          |
|  10   |                    |      |          |

#### 2. 调整learning rate

| 模型 | 准确率（accuracy） | loss | f1 score |
| :--: | :----------------: | :--: | :------: |
| 2e-5 |       88.93%       |      |          |
| 1e-4 |                    |      |          |

#### 3. 调整max_seq_length

| 模型 | 准确率（accuracy） | loss | f1 score |
| :--: | :----------------: | :--: | :------: |
|  64  |                    |      |          |
| 128  |       88.93%       |      |          |
| 256  |                    |      |          |



## 调整训练集大小

| 训练集大小 | 准确率（accuracy） | loss | f1 score |
| :--------: | :----------------: | :--: | :------: |
|   52000    |       88.93%       |      |          |
|   10000    |                    |      |          |
|    5000    |                    |      |          |
|    1000    |                    |      |          |
|    500     |                    |      |          |

未来可能加入不同训练集规模下，与其他模型的性能对比