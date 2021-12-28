from pyspark.sql import SparkSession, Row
from pyspark import SQLContext
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, ArrayType
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler, MinHashLSH
from pyspark.ml.regression import RandomForestRegressor, DecisionTreeRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import sys
import string
import math

pieces = "KQBRN"
files = "abcdefgh"
numbers = "12345678"

min_test = 50

#   a  b  c  d  e  f  g  h
# 8 _  _  _  R  _  _  _  R
# 7 _  _  _  _  _  _  _  _
# 6 _  _  _  _  _  _  _  _
# 5 r  _  _  _  _  _  _  _
# 4 _  _  _  _  p  _  _  p
# 3 _  _  _  _  _  _  _  _
# 2 _  _  _  _  _  _  _  _
# 1 r  _  _  _  _  _  _  q

piece_values = {'_': 0, 'p': 1, 'n': 3, 'r': 5, 'b': 7, 'q': 10, 'k': 12, 'P': 1, 'N': 3, 'R': 5, 'B': 7, 'Q': 10, 'K': 12}

def render_board(board):
    print("\n  a  b  c  d  e  f  g  h ")

    for x in range(8):
        row_chars = board[x*8:(x+1)*8]
        row_str = str(8-x)+" "
        for ch in row_chars:
            row_str+=ch+"  "
        print(row_str)
    print("\n")

def get_move_type(move):
    if move == "O-O-O":
        return "queenside_castle"

    if move == "O-O":
        return "kingside_castle"

    if '=' in move:
        return "promotion"

    if 'x' in move:
        if move[1] == 'x' and move[0] in files:
            return "pawn_capture"
        elif len(move) == 4:
            return "capture"
        else:
            return "complex_capture"

    if len(move) == 2:
        #pawn moving
        return "pawn"
    elif len(move) == 3:
        if move[0] in pieces and move[1] in files and move[2] in numbers:
            return "piece"
    else:
        return "complex"

def set_square(position, character, board):
    col = files.index(position[0])#e->4
    row = 8-int(position[1])
    index = row*8+col
    new_board = board[:index]+character+board[index+1:]
    return new_board

def get_square(position, board):
    col = files.index(position[0])
    row = 8-int(position[1])
    return board[row*8+col]

def castle_queenside_white(board):
    board = set_square("e1", "_", board) #remove king
    board = set_square("h1", "_", board) #remove rook
    board = set_square("g1", "K", board) #place king
    board = set_square("f1", "R", board) #place rook
    return board

def castle_kingside_white(board):
    board = set_square("e1", "_", board) #remove king
    board = set_square("a1", "_", board) #remove rook
    board = set_square("c1", "K", board) #place king
    board = set_square("d1", "R", board) #place rook
    return board

def castle_queenside_black(board):
    board = set_square("e8", "_", board) #remove king
    board = set_square("h8", "_", board) #remove rook
    board = set_square("g8", "K", board) #place king
    board = set_square("f8", "R", board) #place rook
    return board

def castle_kingside_black(board):
    board = set_square("e8", "_", board) #remove king
    board = set_square("a8", "_", board) #remove rook
    board = set_square("c8", "K", board) #place king
    board = set_square("d8", "R", board) #place rook
    return board

def pawn_move(move, player, board):
    #player is 1 for white, 0 for black
    file = move[0]
    destination_rank = int(move[1])
    originating_rank = (destination_rank - 1) if player == 1 else (destination_rank + 1)
    if get_square(file+str(originating_rank), board) == "_": #if the row before the one moving to is empty, we came from 2 behind (first move)
        originating_rank = (originating_rank - 1) if player == 1 else (originating_rank + 1)
    pawn_piece = "P" if player==1 else "p"
    board = set_square(file+str(destination_rank), pawn_piece, board)

    board = set_square(file+str(originating_rank), "_", board)
    return board

def pawn_capture(move, player, board):
    destination_rank = int(move[-1])
    destination_file = move[-2]
    originating_file = move[0]
    originating_rank = (destination_rank - 1) if player == 1 else (destination_rank+1)
    originating_square = originating_file+str(originating_rank)
    board = set_square(destination_file+str(destination_rank), get_square(originating_square, board), board)
    board = set_square(originating_file+str(originating_rank), "_", board)
    return board

def promote(move, player, board):
    board = pawn_move(move[:2], player, board) #successfully finds the correct pawn and moves it to the destination
    #now to swap the destination with the correct piece
    new_piece = lower(move[-1]) if player == 0 else move[-1]
    board = set_square(move[:2], new_piece, board)
    return board

def piece_move(move, player, board):
    piece = move[0]
    originating_location = None
    destination = move[1:]
    if   piece == "N":
        originating_location = find_knight(destination, player, board)
    elif piece == "R":
        originating_location = find_rook(destination, player, board)
    elif piece == "B":
        originating_location = find_bishop(destination, player, board)
    elif piece == "Q":
        originating_location = find_queen(destination, player, board)
    elif piece == "K":
        originating_location = find_king(destination, player, board)
    if originating_location:
        piece = get_square(originating_location, board)
        board = set_square(originating_location, "_", board)
        board = set_square(destination, piece, board)
        return board
    return None

def capture(move, player, board):
    board = piece_move(move[0:1]+move[2:], player, board)
    return board

def complex_move(move, player, board):
    _files = None
    _ranks = None
    if len(move) == 5:
        _files = [move[1]]
        _ranks = [move[2]]
    else:
        if move[1] in files:
            _files = [move[1]]
        else:
            _ranks = [move[1]]
    piece = move[0]
    originating_location = None
    destination = move[-2:]
    if   piece == "K":
        originating_location = find_knight(destination, player, board, _files, _ranks)
    elif piece == "R":
        originating_location = find_rook(destination, player, board, _files, _ranks)
    elif piece == "B":
        originating_location = find_bishop(destination, player, board, _files, _ranks)
    elif piece == "Q":
        originating_location = find_queen(destination, player, board, _files, _ranks)
    elif piece == "K":
        originating_location = find_king(destination, player, board, _files, _ranks)
    if originating_location:
        piece = get_square(originating_location, board)
        board = set_square(originating_location, "_", board)
        board = set_square(destination, piece, board)
        return board
    return None

def complex_capture(move, player, board):
    board = complex_move(move.replace('x', ''), player, board)
    return board

def locate(target, board, _files=None, _ranks=None):
    locations = []
    for f in files:
        for r in numbers:
            if (not _files or f in _files) and (not _ranks or r in _ranks):
                square = f+r
                if target == get_square(square, board):
                    locations.append(square)
    return locations

def find_knight(destination, player, board, _files=None, _ranks=None):
    target = "N" if player == 1 else "n"
    destination_rank = destination[1]
    destination_file = destination[0]
    locations = locate(target, board, _files, _ranks)
    print(f"Locations: {locations}")
    for location in locations:
        rank_diff = abs(int(destination_rank)-int(location[1]))
        file_diff = abs(files.index(destination_file) - files.index(location[0]))
        if (rank_diff == 1 and file_diff == 2) or (rank_diff == 2 and file_diff == 1):
            return location
    return None

def find_rook(destination, player, board, _files=None, _ranks=None):
    target = "R" if player == 1 else "r"
    destination_rank = destination[1]
    destination_file = destination[0]
    locations = locate(target, board, _files, _ranks)
    for location in locations:
        rank_diff = abs(int(destination_rank)-int(location[1]))
        file_diff = abs(files.index(destination_file) - files.index(location[0]))
        if (rank_diff== 0) or (file_diff == 0):
            return location
    return None

def find_bishop(destination, player, board, _files=None, _ranks=None):
    target = "B" if player == 1 else "b"
    destination_rank = destination[1]
    destination_file = destination[0]
    locations = locate(target, board, _files, _ranks)
    for location in locations:
        rank_diff = abs(int(destination_rank)-int(location[1]))
        file_diff = abs(files.index(destination_file) - files.index(location[0]))
        if (rank_diff == file_diff):
            return location
    return None

def find_queen(destination, player, board, _files=None, _ranks=None):
    target = "Q" if player == 1 else "q"
    destination_rank = destination[1]
    destination_file = destination[0]
    locations = locate(target, board, _files, _ranks)
    for location in locations:
        rank_diff = abs(int(destination_rank)-int(location[1]))
        file_diff = abs(files.index(destination_file) - files.index(location[0]))
        if (rank_diff == file_diff) or (rank_diff == 0) or (file_diff == 0):
            return location
    return None

def find_king(destination, player, board, _files=None, _ranks=None):
    target = "K" if player == 1 else "k"
    destination_rank = destination[1]
    destination_file = destination[0]
    locations = locate(target, board, _files, _ranks)
    for location in locations:
        rank_diff = abs(int(destination_rank)-int(location[1]))
        file_diff = abs(files.index(destination_file) - files.index(location[0]))
        if (rank_diff <= 1 and file_diff <= 1):
            return location
    return None

def do_move(move, player, board):
    type = get_move_type(move)
    if   type == "piece":
        board = piece_move(move, player, board)
    elif type == "pawn":
        board = pawn_move(move, player, board)
    elif type == "capture":
        board = capture(move, player, board)
    elif type == "pawn_capture":
        board = pawn_capture(move, player, board)
    elif type == "promotion":
        board = promote(move, player, board)
    elif type == "complex":
        board = complex_move(move, player, board)
    elif type == "complex_capture":
        board = complex_capture(move, player, board)
    elif type == "queenside_castle":
        board = castle_queenside_white(board) if player == 1 else castle_queenside_black(board)
    elif type == "kingside_castle":
        board = castle_kingside_white(board) if player == 1 else castle_kingside_black(board)
    return board
#piece                      normal moves are: piece's uppercase letter, destination square - Be5
#pawn                       pawns don't have a letter - e6
#capture, pawn_capture      captures are notated by x - Bxe5 or exd5
#                           En Passant captures have the pawn's file of departure + x + destination square - might have e.p.
#promotion                  pawn promotion : piece promoted to is indicated at the end of the move - e8Q - sometimes use = (e8=Q)
#kingside/queenside_castle  castling : 0-0 for kingside, 0-0-0 for queenside
#if 2 identical pieces can move to the same square it's identified by specifying the piece's letter and then
#   1. File of departure
#   2. Rank of departure
#   3. Both File & rank


def get_all_states(moves_string):
    if not moves_string:
        return None
    moves = [m for m in moves_string.split() if not "." in m]
    states = []
    board = "rnbqkbnrpppppppp________________________________PPPPPPPPRNBQKBNR" #starting board
    player = 0 #0 is black - we set to black here even though white goes first because we swap in the loop
    for move in moves:
        try:
            player = 0 if player == 1 else 1
            states.append(board)
            board = do_move(move, player, board)
        except:
            print(f"Error in state_parser with move {move}.\nLast recorded states: ")
            print(states)
    return states

def get_saveable_states(moves_string):
    states = get_all_states(moves_string)
    if states:
        return " ".join([s for s in states if s])
    else:
        return ""

def simplify(x):
    if x[0]=='1' or x[0]==1:
        return 1
    else:
        return -1

def count_pieces(board):
    pieces = ["P", "N", "R", "B", "Q", "K", "p", "n", "r", "b", "q", "k"]
    counts = [0 for p in pieces]
    for c in board:
        if c!="_":
            if c not in pieces:
                print(f"UNKNOWN CHARACTER: {c}")
                render_board(board)
            else:
                counts[pieces.index(c)]+=1
    return tuple(counts)

def count_wins(a, b):
    if a == 1 and b == 1:
        return 2
    elif a == 1 or b == 1:
        return 1
    else:
        return 0

def piece_sum(counts):
    pieces = ["P", "N", "R", "B", "Q", "K", "p", "n", "r", "b", "q", "k"]
    #counts is a tuple of the same length
    sum = 0
    for c in range(len(counts)):
        sum+=c*piece_values[pieces[c]]
    return sum

def board_intersection(a, b):
    sum = 0
    for x in range(len(a)):
        if a[x] == b[x]:
            sum+=piece_values[a[x]]
    return sum

def board_similarity(a, b):
    intersection = board_intersection(a, b)
    sum_a = piece_sum(count_pieces(a))
    sum_b = piece_sum(count_pieces(b))
    return (intersection/sum_a) if sum_a > sum_b else (intersection/sum_b)

def piece_tuple(board, piece, expected):
    out = []
    for ch in board:
        if ch == piece:
            out.append(1)
    while len(out) < expected:
        out.append(0)
    return tuple(out)

def get_binary_pieceset(board):
    out = ()
    pieces = ["P", "N", "R", "B", "Q", "K", "p", "n", "r", "b", "q", "k"]
    expected = {"P": 8, "N": 2, "R": 2, "B": 2, "Q": 1, "K": 1}
    for piece in pieces:
        out = out+piece_tuple(board, piece, expected[piece.upper()])
    return out

def get_binary_board(board):
    out = []
    for ch in board:
        if ch == "_":
            out.append(0)
        else:
            out.append(1)
    return out

def get_1_indices(vec):
    out = []
    for i in range(len(vec)):
        if vec[i] == 1:
            out.append(i)
    return out

def get_dense_vec(x):
    return Vectors.dense(x)

def get_sparse_board(board):
    binary_board = get_binary_board(board)
    sparsevec = [ (int(i), 1.0) for i in get_1_indices(binary_board) ]
    print("\n\nSPARSEVEC")
    print(sparsevec)
    return Vectors.sparse(64, sparsevec)

if __name__ == "__main__":
    _set_size = 1
    #Create a spark session and name it
    spark = SparkSession.builder.appName('chess').getOrCreate()
    #Create a spark context in session
    sc = spark.sparkContext
    #Reduce output by only showing me the errors
    sc.setLogLevel("ERROR")
    #SQL Context
    sqlContext = SQLContext(sc)
    #   result | elo_white | elo_black | rating_diff | eco | states
    fields = [('result', StringType()),('elo_white', IntegerType()),('elo_black', IntegerType()),('rating_diff', FloatType()),('eco', StringType()),('states', StringType())]
    schema = StructType([StructField(s, t, False) for (s, t) in fields])
    games = spark.read.csv("gamestates.csv",schema=schema)

    games = games.filter(games['result'] != '1/2-1/2')
    result_func = udf(lambda x: 1 if x == '1-0' else 0, IntegerType())
    games = games.withColumn('result', result_func(games['result']))
    splitter = udf(lambda s: s.split(), ArrayType(StringType()))
    games = games.withColumn('states', splitter(games['states']))
    games = games.withColumn('state', explode(games['states'])).drop('states')

    state_rdd = games.select('result', 'state').rdd.map(lambda x: (x['state'], (x['result'], 1)))
    state_rdd = state_rdd.reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1])).map(lambda x: (x[0], x[1][0], x[1][1]))
    #rdd = board_string, wins, occurrences

    state_schema = StructType([StructField('board', StringType(), False), StructField('piece_vec', ArrayType(IntegerType()), False), StructField('board_vec', ArrayType(IntegerType()), False), StructField('wins', FloatType(), False), StructField('occurrences', IntegerType())])
    appearances_threshold = 3
    board_rdd = state_rdd.map(lambda x: (x[0], get_binary_pieceset(x[0]), get_binary_board(x[0]), float(x[1]), x[2]))
    boards = board_rdd.toDF(schema=state_schema)
    va = VectorAssembler(inputCols = ["piece_vec", "board_vec"], outputCol="features", handleInvalid='skip')
    to_vec = udf(lambda x: get_dense_vec(x), VectorUDT())
    add_vecs = udf( lambda a, b: a+b, ArrayType(IntegerType()))
    boards = boards.withColumn('features', add_vecs(boards['piece_vec'], boards['board_vec']))


    boards = boards.withColumn('winrate', boards['wins']/boards['occurrences'])
    elements = 96 #32 pieces and 64 squares
    assembler = VectorAssembler(inputCols=["elem"+str(i) for i in range(elements)], outputCol="pb_vector")
    get_num = udf(lambda p, b, index: p[index] if index < 32 else b[index-32], IntegerType())
    for i in range(elements):
        boards = boards.withColumn("elem"+str(i), get_num(boards['piece_vec'], boards['board_vec'], lit(i)))
    boards = assembler.transform(boards).select('pb_vector', 'winrate', 'wins', 'occurrences')

    appearances_threshold = 25
    (train, test) = boards.randomSplit([0.8, 0.2])
    uncommon_df = train.filter(train['occurrences'] < appearances_threshold)
    common_df = train.filter(train['occurrences'] > appearances_threshold)
    minhash = MinHashLSH(inputCol='pb_vector', outputCol='hash').fit(common_df)
    common_df = minhash.transform(common_df)
    common_df = minhash.approxSimilarityJoin(common_df, uncommon_df, 0.90)

    #approxSimilarityJoin packs these into nested Structs - this is unpacking them
    for field in ['pb_vector', 'winrate', 'wins', 'occurrences', 'hash']:
        common_df = common_df.withColumn(field, common_df.datasetA[field])
        common_df = common_df.withColumn(field+"_uncommon", common_df.datasetB[field])
    #we unpacked these so we don't need the columns anymore
    commons = common_df.rdd.map(lambda x: (x['pb_vector'],  (x['wins'], x['occurrences'], x['wins_uncommon'], x['occurrences_uncommon']) ) )
    commons = commons.reduceByKey(lambda a, b: (a[0], a[1], a[2]+b[2], a[3]+b[3])) #add uncommons together - leave commons alone
    commons = commons.map(lambda x: (x[0], x[1][0]/x[1][1], x[1][2]/x[1][3],  (x[1][0]+x[1][2])/(x[1][3]+x[1][1]), float(x[1][0]+x[1][2]), float(x[1][3]+x[1][1])))
    #                                 pb    original_wr     #addition           total_wr                                new_wins        new_occ
    commons_df = commons.toDF(schema = StructType([StructField('pb_vector', VectorUDT(), False), \
                                                   StructField('original_wr', FloatType(), False),\
                                                   StructField('added_wr', FloatType(), False), \
                                                   StructField('winrate', FloatType(), False), \
                                                   StructField('wins', FloatType(), False), \
                                                   StructField('occurrences', FloatType(), False)]))


    eval = RegressionEvaluator(predictionCol='prediction', labelCol='winrate', metricName='mae')
    f_test = test.filter(test['occurrences'] > min_test)
    lr = LinearRegression(featuresCol='pb_vector', labelCol='winrate', predictionCol='prediction', maxIter=1000, tol=0.001)
    lr_model = lr.fit(train)
    test_lr = lr_model.transform(f_test)
    print(f"Linear Regression MAE : {eval.evaluate(test_lr)}")
