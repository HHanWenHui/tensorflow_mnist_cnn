# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 20:22:08 2019

@author: DongW
"""

#====加载数据
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist=input_data.read_data_sets("MNIST_data_cnn/",one_hot=True)


#=====构建模型
#输入数据
trX,trY,teX,teY=mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels
#处理数据
trX=trX.reshape(-1,28,28,1)
teX=teX.reshape(-1,28,28,1)
X=tf.placeholder("float",[None,28,28,1])
Y=tf.placeholder("float",[None,10])
#初始化权重
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01))
#patch
w=init_weights([3,3,1,32])
w2=init_weights([3,3,32,64])
w3=init_weights([3,3,64,128])
#全连接层
w4=init_weights([128*4*4,625])
#输出层0
w_o=init_weights([625,10])
#定义模型函数
#神经网络模型的构建函数，传入以下参数:X,w，p_keep_conv,p_keep_hidden
def model(X,w,w2,w3,w4,w_o,p_keep_conv,p_keep_hidden):
    #第一组卷积层+池化层
    l1a=tf.nn.relu(tf.nn.conv2d(X,w,strides=[1,1,1,1],padding='SAME'))
    #?,28,28,32
    l1=tf.nn.max_pool(l1a,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    #?,14,14,32
    l1=tf.nn.dropout(l1,p_keep_conv)
    
     #第二组卷积层+池化层
    l2a=tf.nn.relu(tf.nn.conv2d(l1,w2,strides=[1,1,1,1],padding='SAME'))
    #?,14,14,64
    l2=tf.nn.max_pool(l2a,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    #?,7,7,64
    l2=tf.nn.dropout(l2,p_keep_conv)
    
     #第三组卷积层+池化层
    l3a=tf.nn.relu(tf.nn.conv2d(l2,w3,strides=[1,1,1,1],padding='SAME'))
    #?,7,7,128
    l3=tf.nn.max_pool(l3a,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    #?,4,4,128
    l3=tf.reshape(l3,[-1,w4.get_shape().as_list()[0]])
    #?,2048
    l3=tf.nn.dropout(l3,p_keep_conv)
    
    #全连接层
    l4=tf.nn.relu(tf.matmul(l3,w4))
    l4=tf.nn.dropout(l4,p_keep_hidden)
    
    #输出层
    pyx=tf.matmul(l4,w_o)
    #返回预测值
    return pyx
#定义dropout的占位符   keep_conv
p_keep_conv=tf.placeholder("float")
p_keep_hidden=tf.placeholder("float")
#得到预测值
py_x=model(X,w,w2,w3,w4,w_o,p_keep_conv,p_keep_hidden)
#定义损失函数
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x,labels=Y))
train_op=tf.train.RMSPropOptimizer(0.001,0.9).minimize(cost)
predict_op=tf.argmax(py_x,1)



#=======训练模型和评估模型
#定义训练和评估时的批次大小
batch_size=128
test_size=256
#在一个会话中启动图，开始训练和评估：
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    for i in range(100):
        training_batch=zip(range(0,len(trX),batch_size),
                           range(batch_size,len(trX)+1,batch_size))
        for start,end in training_batch:
            sess.run(train_op,feed_dict={X:trX[start:end],
                                         Y:trY[start:end],
                                         p_keep_conv:0.8,
                                         p_keep_hidden:0.5})
        test_indices=np.arange(len(teX))
        np.random.shuffle(test_indices)
        test_indices=test_indices[0:test_size]
        
        print(i,np.mean(np.argmax(teY[test_indices],axis=1)==
                        sess.run(predict_op,feed_dict={X:teX[test_indices],
                                                       p_keep_conv:1.0,
                                                       p_keep_hidden:1.0})))


























