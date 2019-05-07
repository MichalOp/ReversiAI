import random
import sys
from collections import defaultdict as dd
import tensorflow as tf
import numpy as np
from copy import deepcopy
from queue import Queue
import math
import multiprocessing
import threading
import datetime
#####################################################
RESET = False
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

conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=128,
      kernel_size=[3, 3],
      padding="valid",
      activation=tf.nn.relu)

conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=256,
      kernel_size=[3, 3],
      padding="valid",
      activation=tf.nn.relu)

full_input = tf.reshape(conv2,[-1,4*4*256])

W1 = tf.Variable(tf.random_normal(stddev=0.01,shape = [4*4*256,s1]))#4*4*256
b1 = tf.Variable(tf.random_normal(stddev=0.01,shape = [1,s1]))
W2 = tf.Variable(tf.random_normal(stddev=0.01,shape = [s1,s2]))
b2 = tf.Variable(tf.random_normal(stddev=0.01,shape = [1,s2]))
W3 = tf.Variable(tf.random_normal(stddev=0.01,shape = [s2,M*M]))
b3 = tf.Variable(tf.random_normal(stddev=0.01,shape = [1,M*M]))
W3_value = tf.Variable(tf.random_normal(stddev=0.01,shape = [s2,1]))
b3_value = tf.Variable(tf.random_normal(stddev=0.01,shape = [1,1]))

l1 = tf.nn.relu(tf.matmul(full_input,W1)+b1)
l2 = tf.nn.relu(tf.matmul(l1,W2)+b2)
probs = tf.nn.softmax(tf.matmul(l2,W3)+b3)

value = tf.nn.sigmoid(tf.matmul(l2,W3_value)+b3_value)

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
    #print (convert(board))
    #print (masks)
    data, value = sess.run(joint,feed_dict={x:np.reshape(np.concatenate([convert(board),masks],2),[1,8*8*3])})
    return np.reshape(data,[M,M]), value

#####################################################

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

def copyboard(board):
    return [list(x) for x in board]

def initial_board():
    B = [ [None] * M for i in range(M)]
    B[3][3] = 1
    B[4][4] = 1
    B[3][4] = 0
    B[4][3] = 0
    return B


class Board:
    dirs  = [ (0,1), (1,0), (-1,0), (0,-1), (1,1), (-1,-1), (1,-1), (-1,1) ]


    def __init__(self, board = None):
        if board == None:
            self.board = initial_board()
            self.fields = set()
            self.edge = set()
            self.move_list = []
            for i in range(M):
                for j in range(M):
                    if self.board[i][j] == None:
                        if any(not self.get(i+direction[0],j+direction[1]) == None for direction in Board.dirs):
                            self.edge.add((i,j))
                        self.fields.add( (i,j) )
        else:
            self.board = [list(x) for x in board.board]
            self.fields = set(board.fields)
            self.edge = set(board.edge)
            self.move_list = list(board.move_list)

    def draw(self):
        for i in reversed(range(M)):
            res = []
            for j in range(M):
                b = self.board[i][j]
                if b == None:
                    #if (j,i) in self.edge:
                    #    res.append('x')
                    #else:
                    res.append('.')
                elif b == 1:
                    res.append('#')
                else:
                    res.append('o')
            print (''.join(res))
        print()

    def moves(self, player, debug = False):
        res = []
        for (x,y) in self.edge:
            #if (x,y) == (7,1) and debug:
                #print("here we go with debug")
                #for direction in Board.dirs:
                #    print(direction)
                #    print(self.can_beat(x,y, direction, player))
            if any( self.can_beat(x,y, direction, player) for direction in Board.dirs):
                res.append( (x,y) )
        if not res:
            return [None]
        return res

    def can_beat(self, x,y, d, player):
        dx,dy = d
        x += dx
        y += dy
        cnt = 0

        v = None
        if 0 <= x < M and 0 <=y < M:
            v = self.board[y][x]

        while v == 1-player:
            x += dx
            y += dy
            v = None
            if 0 <= x < M and 0 <=y < M:
                v = self.board[y][x]
            cnt += 1
        return cnt > 0 and self.get(x,y) == player

    def get(self, x,y):
        if 0 <= x < M and 0 <=y < M:
            return self.board[y][x]
        return None

    def do_move(self, move, player):
        #self.history.append([x[:] for x in self.board])
        self.move_list.append(move)

        if move == None:
            return
        x,y = move
        x0,y0 = move
        self.board[y][x] = player
        self.fields -= set([move])
        self.edge -= set([move])
        for dx,dy in self.dirs:
            x,y = x0,y0
            to_beat = []
            x += dx
            y += dy
            if (x,y) in self.fields:
                self.edge.add((x,y))
            while self.get(x,y) == 1-player:
              to_beat.append( (x,y) )
              x += dx
              y += dy
            if self.get(x,y) == player:
                for (nx,ny) in to_beat:
                    self.board[ny][nx] = player

    def result(self):
        res = 0
        for y in range(M):
            for x in range(M):
                b = self.board[y][x]
                if b == 0:
                    res -= 1
                elif b == 1:
                    res += 1
        return res

    def terminal(self):
        if not self.fields:
            return True
        if len(self.move_list) < 2:
            return False
        return self.move_list[-1] == self.move_list[-2] == None

    def random_move(self, player):
        ms = self.moves(player)
        if ms:
            return random.choice(ms)
        return [None]

player = 0
B = Board()

choose_best_prob = 0.9

class Node:
    def __init__(self, player, board, move_probs, value):
        self.player = player
        self.visits = 1
        self.W = 0
        self.value = value
        self.q = value
        self.board = board
        self.move_probs = move_probs
        self.moves = {}

    def expand(self,move,node):
        self.moves[move] = node

    def stepdown(self,move):
        if move in self.moves:
            return self.moves[move]
        else:
            return None


def generate_playable_fields(moves):
    fields = np.ndarray([M,M,1])
    fields.fill(0)
    if not moves[0] == None:
        for move in moves:
            #print(move)
            fields[move[0]][move[1]][0] = 1
    return fields

print(run_net(B.board, generate_playable_fields(B.moves(0))))
'''
def run_net(board):
    board = deepcopy(board)
    for i in range(M):
        for j in range(M):
            board[i][j] = 0.5
    return board
'''

tries = 500

def reverse_board(board):
    for i in range(M):
        for j in range(M):
            if not board[i][j]== None:
                board[i][j] = 1-board[i][j]



def generateMoveProbs(board, player):
    moves = board.moves(player)
    b = copyboard(board.board)
    if player == 1:
        reverse_board(b)
    probs, value = run_net_async(b,generate_playable_fields(moves))
    #print(probs)
    #print(value)
    ps = []
    best = 0
    highest = 0
    if moves == [None]:
        ps.append(1)
    else:
        for move in moves:
            ps.append(float(probs[move[0]][move[1]]))
    #ps = np.array(ps)
    return (moves,ps), value

'''
oooo###o
oooooooo
o###ooo#
o#oooo##
ooooo#.#
ooooo###
##o#####
########
'''

def pass_tree(node):
    #node.board.draw()
    node.visits+=1
    moves, probs = node.move_probs

    #probs = list(probs)
    #print(probs)
    probs = np.array(probs) + 0.03

    for i in range(len(moves)):
        if moves[i] in node.moves:
            child = node.moves[moves[i]]
            probs[i] = probs[i] *math.sqrt(node.visits)/(child.visits+1) + 2*child.q-1
        else:
            probs[i] = probs[i]*math.sqrt(node.visits)

    target = np.argmax(probs)
    move = moves[target]
    new_node = None
    if moves[target] in node.moves:
        new_node = node.moves[moves[target]]


    value = 0

    if new_node == None and not node.board.terminal():
        new_board = Board(node.board)
        new_board.do_move(move,node.player)
        probs, value = generateMoveProbs(new_board,1-node.player)
        value = float(value)
        new_node = Node(1-node.player,new_board,probs,value)
        new_node.W = 1 - new_node.value
        new_node.q = 1 - new_node.value
        node.expand(move,new_node)

        value = 1 - new_node.value

    else:
        if node.board.terminal():
            if (node.board.result()<0 and node.player == 0) or (node.board.result()>0 and node.player == 1):
                value = 1

            else:
                value = 0
        else:
            value = 1-pass_tree(new_node)

    node.W += 1 - value
    node.q = node.W/node.visits

    return value



def monte_carlo_search(root, randomness):

    for i in range(tries):
        pass_tree(root)

    node = root

    node.visits+=1
    moves, probs = node.move_probs
    #print(np.array(probs))
    probs = []

    for i in range(len(moves)):
        child = node.stepdown(moves[i])
        val = 0
        if not child == None:
            val = child.visits

        probs.append(val)

    #print(probs)
    probs = probs/np.sum(probs)

    probs_board = np.ndarray([M,M])
    probs_board.fill(0)
    if not moves[0] == None:
        for i in range(len(moves)):
            probs_board[moves[i][0]][moves[i][1]] = probs[i]

    index = 0
    if not randomness:
        index = np.argmax(probs)
    else:
        index = np.random.choice(range(len(probs)),1,p = probs)[0]
    move = moves[index]
    new_node = node.stepdown(move)

    return move, new_node, node.board.board, probs_board, node.player

total_dataset = []
'''
o#oooooo
o##ooooo
o#o#o#oo
oo#oo#oo
oo#o###o
oo#####o
oo####oo
#.#####o
'''

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

def run_test_game_with_montecarlo():
    board = Board()
    probs, value = generateMoveProbs(board,0)
    value = float(value)
    #print(value)
    root = Node(0,board,probs, value)
    root.q = root.value

    player = 0

    while not board.terminal():

        board.draw()

        if player == 0:

            print(root.value)
            print(root.q)
            move, new_node, current_board, monte_carlo_probs, current_player = monte_carlo_search(root,True)

            board.do_move(move, player)
            root = new_node
            #player = 1-player

        else:
            move = board.random_move(player)
            root = root.stepdown(move)
            board.do_move(move,player)
        player = 1-player
    board.draw()
    print(board.result())
    return board.result()<0

game_stories_queue = Queue()
def run_train_game():
    board = Board()
    probs, value = generateMoveProbs(board,0)
    value = float(value)
    #print(value)
    root = Node(0,board,probs, value)
    root.q = root.value

    _game_story = []

    iteration = 0

    time_old = datetime.datetime.now()

    while not board.terminal():
        #print(str(threading.current_thread().name))
        iteration+=1
        #board.draw()
        #print(root.value)
        #print(root.q)
        fields = generate_playable_fields(board.moves(root.player))
        move, new_node, current_board, monte_carlo_probs, current_player = monte_carlo_search(root,iteration<15)

        b = copyboard(current_board)
        if root.player == 1:
            reverse_board(b)
        x, y = run_net(b,fields)
        #print(x * np.reshape(fields,[M,M]) - 1 + np.reshape(fields,[M,M]))
        #print(current_player)
        _game_story.append((copyboard(current_board),fields,monte_carlo_probs,current_player))
        _game_story.append((flip_board(current_board),flip_board(fields),flip_board(monte_carlo_probs),current_player))
        _game_story.append((flip_board2(current_board),flip_board2(fields),flip_board2(monte_carlo_probs),current_player))
        _game_story.append((rotate_board(current_board),rotate_board(fields),rotate_board(monte_carlo_probs),current_player))
        board.do_move(move, root.player)
        root = new_node

        time_now = datetime.datetime.now()
        print(time_now - time_old)
        time_old = time_now

    winner = 0
    if board.result()<0:
        winner = 1

    game_story = []
    for c,f,m,p in _game_story:
        if p == 1:
            reverse_board(c)
        #for l in c:
        #    print(c)
        game_story.append((c,f,m,abs(winner-p)))

    global game_stories_queue

    game_stories_queue.put(game_story)



def run_game_against_me():
    board = Board()
    player = 0
    count = 0

    while not board.terminal():

        m = None
        if player == 1:
            moves = board.moves(player)
            b = deepcopy(board.board)
            reverse_board(b)
            probs, val = run_net(b,generate_playable_fields(moves),live_sess)
            best = None
            highest = -1
            ps = []
            if not moves == [None]:
                #print(board.board)
                for move in moves:
                    ps.append(probs[move[0]][move[1]])
                    if probs[move[0]][move[1]] > highest:

                        highest = probs[move[0]][move[1]]
                        best = move

            if count>=10:
                m = best
            else:
                if len(moves) == 1:
                    m = moves[0]
                else:
                    ps = np.array(ps)
                    ps = ps/np.sum(ps)
                    m = moves[np.random.choice(range(len(moves)),p=ps)]
        else:
            moves = board.moves(player)
            probs, val = run_net(board.board,generate_playable_fields(moves))
            ps = []
            best = None
            highest = -1
            if not moves == [None]:
                #print(board.board)
                for move in moves:
                    ps.append(probs[move[0]][move[1]])
                    if probs[move[0]][move[1]] > highest:

                        highest = probs[move[0]][move[1]]
                        best = move

            if count>10:
                m = best
            else:
                if len(moves) == 1:
                    m = moves[0]
                else:
                    ps = np.array(ps)
                    ps = ps/np.sum(ps)
                    m = moves[np.random.choice(range(len(moves)),p=ps)]

        board.do_move(m,player)
        player = 1-player
        count+=1

    #print(board.result())
    return board.result() < 0

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
    if len(total_dataset)>50000:
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

'''
wins = 0
for i in range(1000):
    bo = deepcopy(B)
    if monte_carlo_pass(bo,0):
        #print(wins)
        wins+=1

print(wins)
'''

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
'''

wins = 0

for i in range(100):
    if run_test_game_with_montecarlo():
        print("win")
        wins+=1

print("#########################################################")
print(wins)
with open("wins",'w') as f:
    f.write(str(wins)+'\n')
print("#########################################################")
'''
thread = threading.Thread(target = solver)
thread.start()

while True:

    #do_train_step(tasks.get())
    for i in range(3):
        train_full(Board())
    #if run%500 == 0:
    test()

'''
while True:
    B.draw()
    #B.show()
    m = B.random_move(player)
    B.do_move(m, player)
    player = 1-player
    #input()
    if B.terminal():
        break

B.draw()
#B.show()
print ('Result '+ str(B.result()))
print('Game over!')
'''

sys.exit(0)
