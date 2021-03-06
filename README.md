# 问题摘要与推理（NLG）
1、数据集：
  
-- 训练集（82943条记录）建立模型，基于汽车品牌、车系、问题内容与问答对话的文本，输出建议报告文本。

-- 测试集（20000条记录）使用训练好的模型，输出建议报告的结果文件。

# 项目文件说明

## 1、word2vec文件夹下主要是NLG第一阶段的任务，进行词向量训练

* 主要任务：提取/整理有用的数据、整合成标准的训练数据/测试数据、生成词典、训练词向量等

## 2、baseline文件夹主要是NLG的baseline,主要用的seq2seq模型，其中会包含word2vec的代码，使项目层层递进方式进行
* 模型：encode采用双向gru,decode采用gru的seq2seq结构；外加attention结构
* 训练方式：TeacherForcing
* 预测方式：beamSearch
* 存在缺陷：出现OOV问题、预测出现大量重复词

## 3、notebook文件夹下存放平时代码练习与代码总结
## 4、doc文件夹下存放平时感悟与总结
## 5、utils文件夹存放工具类
## 6、tf_pgn文件夹基于baseline的改进
* pgn

  -- rnn_decoder.py新增Pointer类，计算pgen系数
 
  -- pgn.py中记录每一次的pgen

  -- decoding.py中将attention概率与预测概率融合

* coverage

  -- rnn_decoder.py中attention类进行修改，详见BahdanauAttentionCoverage
 
  -- pgn.py中记录每一次的attention

  -- loss.py中计算coverage_loss
  
## 7、transformer_pgn文件夹采用transformer+pgn结构

* transformer+pgn
  
  -- models/transformer.py中修改Decoder,在最后一层输出与attention计算pgen

  -- models/transformer.py中PGN_TRANSFORMER中调用decoding.py中方法，将attention概率与预测概率融合
