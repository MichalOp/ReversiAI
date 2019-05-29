import random
import numpy as np
from queue import Queue
import math
import multiprocessing
import datetime
#####################################################
RESET = True
BOK = 30
SX = -100
SY = 0
M = 8
batch = 64

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3

sess = tf.Session(config = config)
live_sess = tf.Session(config = config)

saver = tf.train.Saver()

if RESET:
    saver.save(sess,'./Models/model.ckpt')
else:
    saver.restore(sess,'./Models/model.ckpt')

live_sess.run(tf.global_variables_initializer())
saver.restore(live_sess,'./Models/model.ckpt')


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

def flip_board(board):
        new_board = deepcopy(board)
        for i in range(M):
            for j in range(M):
                new_board[i][j] = board[j][i]

        return new_board

    def flip_board2(board):
        new_board = deepcopy(board)
        for i in range(M):
            for j in range(M):
                new_board[i][j] = board[M-j-1][M-i-1]

        return new_board

    def rotate_board(board):
        new_board = deepcopy(board)
        for i in range(M):
            for j in range(M):
                new_board[i][j] = board[M-i-1][M-j-1]

        return new_board

def train(boards, ys,values, masks,sess = sess):
    masks = np.concatenate([np.reshape(m,[1,M,M,1])for m in masks])
    target = np.concatenate([np.reshape(y,[1,M*M]) for y in ys])
    boards = np.concatenate([np.concatenate([np.reshape(convert(b),[1,M,M,2]) for b in boards]),masks],3)
    #print(boards)
    values = np.concatenate([np.reshape(np.array(m),[1,1]) for m in values])
    sess.run(train_opt,feed_dict={x:np.reshape(boards,[-1,M*M*3]),value_target:values, y_target:target})

def return_loss(boards, ys, values, masks, sess=sess):
    masks = np.concatenate([np.reshape(m,[1,M,M,1])for m in masks])
    target = np.concatenate([np.reshape(y,[1,M*M]) for y in ys])
    boards = np.concatenate([np.concatenate([np.reshape(convert(b),[1,M,M,2]) for b in boards]),masks],3)
    #print(boards)
    values = np.concatenate([np.reshape(np.array(m),[1,1]) for m in values])
    return sess.run(true_loss,feed_dict={x:np.reshape(boards,[-1,M*M*3]),value_target:values, y_target:target})

network_tasks = Queue()
answer_dictionary = {}
process_flag = threading.Event()

def run_net_async(board,masks):

    lock_flag = threading.Event()
    global network_tasks
    network_tasks.put((lock_flag,threading.current_thread(),board,masks))
    process_flag.set()
    lock_flag.wait()
    global answer_dictionary
    out = answer_dictionary[threading.current_thread()]
    del answer_dictionary[threading.current_thread()]
    return out


def solver():

    global network_tasks
    global answer_dictionary
    global process_flag

    boards = []
    masks = []

    while True:

        process_flag.wait()

        boards = []
        masks = []
        locks = []
        threads = []

        while not network_tasks.empty():
            l,t,b,m = network_tasks.get()
            boards.append(b)
            masks.append(m)
            locks.append(l)
            threads.append(t)

        if len(masks)>0:
            #print("gpu launch with "+str(len(masks))+" boards")
            masks = np.concatenate([np.reshape(m,[1,M,M,1])for m in masks])
            boards = np.concatenate([np.concatenate([np.reshape(convert(b),[1,M,M,2]) for b in boards]),masks],3)

            probs, values = live_sess.run(joint,feed_dict={x:np.reshape(boards,[-1,M*M*3])})

            for i in range(len(threads)):
                answer_dictionary[threads[i]] = (np.reshape(probs[i],[M,M]),values[i])
                locks[i].set()

        if network_tasks.empty():
            process_flag.clear()


def run_net(board, masks,sess = sess):
    data, value = sess.run(joint,feed_dict={x:np.reshape(np.concatenate([convert(board),masks],2),[1,8*8*3])})
    return np.reshape(data,[M,M]), value

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

player = 0
B = Board()

def test_training():
    co = 0
    for i in range(1000):
        if i%10 == 0:
            print(i/10)
        if run_game_against_me():
            co+=1
    print(co)
    return co>550

total_runs = 0

def train_full(board):
    #test_training()
    global total_dataset
    global game_stories_queue
    #import cProfile
    #cProfile.run('run_train_game()')

    for i in range(3):
        time_old = datetime.datetime.now()
        threads = []
        for i in range(8):
            print("launching simulation thread "+str(i))
            thread = threading.Thread(target = run_train_game)
            thread.start()
            threads.append(thread)


        for i in range(len(threads)):
            print("awaiting thread "+str(i))
            threads[i].join()

        while not game_stories_queue.empty():
            total_dataset += game_stories_queue.get()
        print("COLLECT TIME: "+ str(datetime.datetime.now()-time_old))

    print(f"DATASET_SIZE:{len(total_dataset)}")
    while len(total_dataset)>500000:
        del total_dataset[0]
    if len(total_dataset)>15000:
        for i in range(min(max(100,len(total_dataset)//100),2000)):
            print(i)
            arr = random.sample(total_dataset,min(batch,len(total_dataset)))

            bs = []
            fs = []
            ms = []
            vs = []

            for b,f,m,v in arr:
                bs.append(b)
                fs.append(f)
                ms.append(m)
                vs.append(v)

            #print(fs)
            print(return_loss(bs,ms,vs,fs))
            train(bs,ms,vs,fs)
    global total_runs
    total_runs+=1

    #if test_training() or total_runs<3:
    print("UPDATING")
    saver.save(sess,'./Models/model.ckpt')
    saver.restore(live_sess,'./Models/model.ckpt')


train_list = []

def run_game():
    board = Board()
    player = 0
    while not board.terminal():

        m = None
        if player == 1:
            m = board.random_move(player)
        else:
            moves = board.moves(player)
            probs, val = run_net(board.board,generate_playable_fields(moves),live_sess)
            weighted_moves = []
            best = None
            highest = -1
            if not moves == [None]:
                #print(board.board)
                for move in moves:
                    if probs[move[0]][move[1]] > highest:
                        highest = probs[move[0]][move[1]]
                        best = move
            m = best

        board.do_move(m,player)
        player = 1-player

    #print(board.result())
    return board.result() < 0

def test():
    co = 0
    for i in range(1000):
        if i%10 == 0:
            print(i/10)
        if run_game():
            co+=1
    print('wins: '+str(co))
    with open("log",'a') as f:
        f.write(str(co)+'\n')

thread = threading.Thread(target = solver)
thread.start()

while True:

    #do_train_step(tasks.get())
    for i in range(3):
        train_full(Board())
    #if run%500 == 0:
    test()
