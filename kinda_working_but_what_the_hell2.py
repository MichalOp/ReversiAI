import random
import sys
from collections import defaultdict as dd
import numpy as np
from copy import deepcopy
from queue import Queue
import math
import multiprocessing as mp
import datetime
from time import sleep
#####################################################

RESET = True
BOK = 30
SX = -100
SY = 0
M = 8
batch = 64
tries = 500
choose_best_prob = 0.9

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
            if (x,y) == (7,1) and debug:
                print("here we go with debug")
                for direction in Board.dirs:
                    print(direction)
                    print(self.can_beat(x,y, direction, player))
            if any( self.can_beat(x,y, direction, player) for direction in Board.dirs):
                res.append((x,y))
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
            v = self.board[x][y]

        while v == 1-player:
            x += dx
            y += dy
            v = None
            if 0 <= x < M and 0 <=y < M:
                v = self.board[x][y]
            cnt += 1
        return cnt > 0 and self.get(x,y) == player

    def get(self, x,y):
        if 0 <= x < M and 0 <=y < M:
            return self.board[x][y]
        return None

    def do_move(self, move, player):
        #self.history.append([x[:] for x in self.board])
        self.move_list.append(move)

        if move == None:
            return
        x,y = move
        x0,y0 = move
        self.board[x][y] = player
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
                    self.board[nx][ny] = player

    def result(self):
        res = 0
        for y in range(M):
            for x in range(M):
                b = self.board[x][y]
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

def reverse_board(board):
    for i in range(M):
        for j in range(M):
            if not board[i][j]== None:
                board[i][j] = 1-board[i][j]

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

class Worker:
    
    def __init__ (self, worker_id, task_queue, result_pipe, data_queue, run_flag):
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.result_pipe = result_pipe
        self.data_queue = data_queue
        self.run_flag = run_flag
        
        if not self.run_flag == None:
            self.run()
            
    def run(self):
        while self.run_flag.value != 0:
            if self.run_flag.value == 1:
                self.run_train_game()
            else:
                sleep(0.1)
    
    def run_net_async(self,board,masks):

        self.task_queue.put((self.worker_id,board,masks))
        return self.result_pipe.recv()

    def generate_playable_fields(self,moves):
        fields = np.ndarray([M,M,1])
        fields.fill(0)
        if not moves[0] == None:
            for move in moves:
                #print(move)
                fields[move[0]][move[1]][0] = 1
        return fields

    def generateMoveProbs(self,board, player, debug = False):
        moves = board.moves(player)
        b = copyboard(board.board)
        if player == 1:
            reverse_board(b)
        
        if debug:
            playable = self.generate_playable_fields(moves)
            for i in range(M):
                res = ""
                for j in range(M):
                    b_ = b[i][j]
                    p = playable[i][j][0]
                    if not b_ == None and p == 1:
                        res += '?'
                    elif p == 1:
                        res += 'p'
                    elif b_ == None:
                        res += '.'
                    elif b_ == 1:
                        res += '#'
                    else:
                        res += 'o'
                print(res)
        
        probs, value = self.run_net_async(b,self.generate_playable_fields(moves))
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

    def pass_tree(self,node):
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
            probs, value = self.generateMoveProbs(new_board,1-node.player)
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
                value = 1-self.pass_tree(new_node)

        node.W += 1 - value
        node.q = node.W/node.visits

        return value

    def monte_carlo_search(self, root, randomness):

        for i in range(tries):
            self.pass_tree(root)

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

    def run_test_game_with_montecarlo(self):
        board = Board()
        probs, value = self.generateMoveProbs(board,0)
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
                move, new_node, current_board, monte_carlo_probs, current_player = self.monte_carlo_search(root,True)

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
    
    def run_train_game(self):
        board = Board()
        probs, value = self.generateMoveProbs(board,0)
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
            self.generateMoveProbs(board,root.player,True)
            #board.draw()
            #print(root.value)
            #print(root.q)
            fields = self.generate_playable_fields(board.moves(root.player))
            move, new_node, current_board, monte_carlo_probs, current_player = self.monte_carlo_search(root,iteration<15)

            b = copyboard(current_board)
            if root.player == 1:
                reverse_board(b)
            #x, y = run_net(b,fields)
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

        self.data_queue.put(game_story)
        
    def run_game_against_me(self):
        board = Board()
        player = 0
        count = 0
        while not board.terminal():
            m = None
            if player == 1:
                moves = board.moves(player)
                b = deepcopy(board.board)
                reverse_board(b)
                probs, val = run_net_async(b,generate_playable_fields(moves))
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

'''
wins = 0
for i in range(1000):
    bo = deepcopy(B)
    if monte_carlo_pass(bo,0):
        #print(wins)
        wins+=1

print(wins)
'''

def test(net):
    co = 0
    for i in range(1000):
        if i%10 == 0:
            print(i/10)
        if run_game(net):
            co+=1
    print('wins: '+str(co))
    with open("log",'a') as f:
        f.write(str(co)+'\n')

def run_worker(worker_id, task_queue, result_pipe, data_queue, run_flag):
    w = Worker(worker_id, task_queue, result_pipe, data_queue, run_flag)

def generate_playable_fields(moves):
    fields = np.ndarray([M,M,1])
    fields.fill(0)
    if not moves[0] == None:
        for move in moves:
            #print(move)
            fields[move[0]][move[1]][0] = 1
    return fields

def run_game(net):
    board = Board()
    player = 0
    while not board.terminal():
        
        m = None
        if player == 1:
            m = board.random_move(player)
        else:
            moves = board.moves(player)
            probs, val = net.run_net(board.board,generate_playable_fields(moves))
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

def init_workers(n):
    worker_flag = mp.Value('i', 1)
    
    task_queue = mp.Queue()
    data_queue = mp.Queue()
    
    workers = []
    for i in range(n):
        end_a, end_b = mp.Pipe()
        workers.append(end_a)
        mp.Process(target=run_worker, args=(i,task_queue,end_b,data_queue,worker_flag)).start()
                
    return task_queue, data_queue, workers,worker_flag



def train():
    
    task_queue,data_queue,workers,worker_flag = init_workers(24)
    
    import network_basic as net
    #test(net)
    total_runs = 0
    total_dataset = []
    while True:

        for i in range(3):
            collected = 0
            
            time_old = datetime.datetime.now()
            while collected < 24:
                
                for v in range(1000):
                    net.solver(workers,task_queue)
                
                while not data_queue.empty():
                    collected += 1
                    total_dataset += data_queue.get()
                    sleep(0.01)
                sleep(0.1)
                
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
                    print(net.return_loss(bs,ms,vs,fs))
                    net.train(bs,ms,vs,fs)
            total_runs+=1

            #if test_training() or total_runs<3:
            print("UPDATING")
            net.saver.save(net.sess,'./Models/model.ckpt')
            net.saver.restore(net.live_sess,'./Models/model.ckpt')

        #if run%500 == 0:
        test(net)

train()
