# 问题摘要与推理（NLG）
1、数据集：
  
--训练集（82943条记录）建立模型，基于汽车品牌、车系、问题内容与问答对话的文本，输出建议报告文本。

--测试集（20000条记录）使用训练好的模型，输出建议报告的结果文件。

#项目文件说明
1、1_word2vec文件下主要是NLG第一阶段的任务，进行词向量训练
2、2_baseline文件主要是NLG的baseline,主要用的seq2seq模型，其中会包含1_word2vec的代码，使项目层层递进方式进行
