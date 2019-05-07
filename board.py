
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
