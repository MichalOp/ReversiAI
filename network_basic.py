import random
import sys
import tensorflow as tf
import numpy as np
import math
import datetime
from time import sleep
#####################################################
RESET = True
BOK = 30
SX = -100
SY = 0
M = 8
batch = 64

#####################################################

s1 = 600
s2 = 600

with open("log",'a') as f:
    f.write(str('---')+'\n')
    
x = tf.placeholder(tf.float32,[None,M*M*3])
input_layer = tf.reshape(x, [-1,8,8,3])

result = tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu)(input_layer)

for i in range(6):
    result = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)(result)


conv_reduce = tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu)(result)

conv_reduce2 = tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=[3, 3],
    padding="valid",
    activation=tf.nn.relu)(conv_reduce)

conv_reduce3 = tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=[3, 3],
    padding="valid",
    activation=tf.nn.relu)(conv_reduce2)

conv_reduce4 = tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=[3, 3],
    padding="valid",
    activation=tf.nn.relu)(conv_reduce3)


conv_probs = tf.keras.layers.Conv2D(
    filters=1,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu)(result)

probs = tf.nn.softmax(tf.keras.layers.Flatten()(conv_probs))

intermediate = tf.keras.layers.Dense(128,activation=tf.nn.relu)(tf.keras.layers.Flatten()(conv_reduce4))
value = tf.keras.layers.Dense(1,activation=tf.nn.sigmoid)(intermediate)

joint = tf.tuple([probs,value])

y_target = tf.placeholder(tf.float32,[None,M*M])
value_target = tf.placeholder(tf.float32,[None,1])
train_target = tf.placeholder(tf.float32,[None,M*M])

true_loss = tf.nn.l2_loss(y_target-probs) + tf.nn.l2_loss(value-value_target)

loss = true_loss + tf.reduce_sum([tf.nn.l2_loss(x)for x in tf.global_variables()])*0.0001

opt = tf.train.AdamOptimizer(0.0003)
train_opt = opt.minimize(loss)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3

sess = tf.Session(config = config)
live_sess = tf.Session(config = config)

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

if RESET:
    saver.save(sess,'./Models/model.ckpt')
else:
    saver.restore(sess,'./Models/model.ckpt')

live_sess.run(tf.global_variables_initializer())
saver.restore(live_sess,'./Models/model.ckpt')

#####################################################
def convert(board):
    out = np.ndarray([M,M,2])
    #print(out)
    out.fill(0)
    for i in range(M):
        for j in range(M):
            if board[i][j] == 0:
                out[i][j][0] = 1
            if board[i][j] == 1:
                out[i][j][1] = 1
    return np.reshape(out,[M,M,2])

def train(boards, ys,values, masks,sess = sess):
    masks = np.concatenate([np.reshape(m,[1,M,M,1])for m in masks])
    target = np.concatenate([np.reshape(y,[1,M*M]) for y in ys])
    boards = np.concatenate([np.concatenate([np.reshape(convert(b),[1,M,M,2]) for b in boards]),masks],3)
    #print(boards)
    values = np.concatenate([np.reshape(np.array(m),[1,1]) for m in values])
    sess.run(train_opt,feed_dict={x:np.reshape(boards,[-1,M*M*3]),value_target:values, y_target:target})

def return_loss(boards, ys,values, masks, sess=sess):
    masks = np.concatenate([np.reshape(m,[1,M,M,1])for m in masks])
    target = np.concatenate([np.reshape(y,[1,M*M]) for y in ys])
    boards = np.concatenate([np.concatenate([np.reshape(convert(b),[1,M,M,2]) for b in boards]),masks],3)
    #print(boards)
    values = np.concatenate([np.reshape(np.array(m),[1,1]) for m in values])
    return sess.run(true_loss,feed_dict={x:np.reshape(boards,[-1,M*M*3]),value_target:values, y_target:target})

def solver(worker_list, task_queue):

    boards = []
    masks = []
    indices = []

    while not task_queue.empty():
        index,b,m = task_queue.get()
        boards.append(b)
        masks.append(m)
        indices.append(index)

    if len(masks)>0:
        #print("gpu launch with "+str(len(masks))+" boards")
        masks = np.concatenate([np.reshape(m,[1,M,M,1])for m in masks])
        boards = np.concatenate([np.concatenate([np.reshape(convert(b),[1,M,M,2]) for b in boards]),masks],3)

        probs, values = live_sess.run(joint,feed_dict={x:np.reshape(boards,[-1,M*M*3])})

        for i in range(len(indices)):
            worker_list[indices[i]].send((np.reshape(probs[i],[M,M]),values[i]))
        
    sleep(0.00002)

def run_net(board, masks):
    data, value = live_sess.run(joint,feed_dict={x:np.reshape(np.concatenate([convert(board),masks],2),[1,8*8*3])})
    return np.reshape(data,[M,M]), value
