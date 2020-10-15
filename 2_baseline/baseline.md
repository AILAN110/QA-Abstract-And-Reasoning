# BASELINE
## 1.文件结构
    * utils是w2v训练词向量的相关函数与工具
    * datas存放的w2v阶段处理好的训练数据、测试数据、字典等等
    * seq2seq_tf2则是模型的整体代码
## 2. baseline采用的是seq2seq的模型
    * encoder采用双向的gru结构
    * attetion计算score的方式，采用MLP来进行融合求score
    * decoder采用单向的gru结构+fc

## 3. 训练方式
   * 采用TeacherForcing的方式进行训练，解决训练时不易收敛，偏差大等问题

## 4. 预测方式
   * 1、采用greedy_search的方式预测句子（一般方式）
   * 2、采用beam_search的方式预测句子（减少TeacherForcing方式训练产生的曝光偏差）



