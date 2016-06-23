共有RNN、BRNN、LSTM和BLSTM四个模型以及它们与CRF的组合模型。
DoubleMax.java 定义了矩阵操作运算和其他一些运算
PropReader读取conf文件夹下的配置文件
Valuate.Java是对结果的评价
运行时： java RNNChunk conf/rnn.properties >> res/test1
输入的数据是词向量，预测的标签表示为37维的one-hot向量