from board import Board, copyboard

tries = 500
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
    
    def generate_playable_fields(self):
        fields = np.ndarray([M,M,1])
        fields.fill(0)
        if not self.moves[0] == None:
            for move in self.moves:
                #print(move)
                fields[move[0]][move[1]][0] = 1
        return fields
    
    def expand(self,move,node):
        self.moves[move] = node

    def stepdown(self,move):
        if move in self.moves:
            return self.moves[move]
        else:
            return None

def reverse_board(board):
    for i in range(M):
        for j in range(M):
            if not board[i][j]== None:
                board[i][j] = 1-board[i][j]

class Worker:
    
    def __init__(self, worker_id, inpipe, task_queue, result_queue, status_flag):
        self.worker_id = worker_id
        self.inpipe = inpipe
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        while status_flag.value != 0:
            if status_flag.value == 1:
                run_train_game()
            else:
                sleep(1000)

    def run_net_async(self, board, fields):
        self.task_queue.put((self.worker_id, board, fields))

    def generateMoveProbs(self,board, player):
        moves = board.moves(player)
        b = copyboard(board.board)
        if player == 1:
            reverse_board(b)
        probs, value = run_net_async(b,b.generate_playable_fields())
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

    def monte_carlo_search(self, root, randomness):

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

    def run_test_game_with_montecarlo(self):
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


    def run_train_game(self):
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
            board.do_move(move, root.player)
            root = new_node

            time_now = datetime.datetime.now()
            print(time_now - time_old)
            time_old = time_now

        winner = 0
        if board.result()<0:
            winner = 1

        for c,f,m,p in _game_story:
            if p == 1:
                reverse_board(c)
            self.result_queue.put((c,f,m,abs(winner-p)))
