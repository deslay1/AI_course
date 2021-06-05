import gym
import random
import requests
import numpy as np
import pdb
import time
import sys, getopt, cmd

from evaluation import utility
from gym_connect_four import ConnectFourEnv

env: ConnectFourEnv = gym.make("ConnectFour-v0")

# SERVER_ADDRESS = "http://localhost:8000/"
SERVER_ADDRESS = "https://vilde.cs.lth.se/edap01-4inarow/"
API_KEY = 'nyckel'
STIL_ID = ["os5222el-s"]

def call_server(move):
   res = requests.post(SERVER_ADDRESS + "move",
                       data={
                           "stil_id": STIL_ID,
                           "move": move, # -1 signals the system to start a new game. any running game is counted as a loss
                           "api_key": API_KEY,
                       })
   # For safety some respose checking is done here
   if res.status_code != 200:
      print("Server gave a bad response, error code={}".format(res.status_code))
      exit()
   if not res.json()['status']:
      print("Server returned a bad status. Return message: ")
      print(res.json()['msg'])
      exit()
   return res

"""
You can make your code work against this simple random agent
before playing against the server.
It returns a move 0-6 or -1 if it could not make a move.
To check your code for better performance, change this code to
use your own algorithm for selecting actions too
"""
def opponents_move(env, depth_lim, random_bot):
   env.change_player() # change to oppoent
   avmoves = env.available_moves()
   if not avmoves:
      env.change_player() # change back to student before returning
      return -1

   # TODO: Optional? change this to select actions with your policy too
   # that way you get way more interesting games, and you can see if starting
   # is enough to guarrantee a win
   if random_bot == "random": 
      action = random.choice(list(avmoves))
   elif random_bot == 'ai':
      state = env.board
      result, action = minimax(env, depth_lim, float('-inf'), float('inf'), True, player_disc = -1)
      env.reset(state)
      if action == -1: action = np.random.choice(list(avmoves))
      env.change_player()
   else:
      action = int(input("Enter a move from 1-6\n"))
   
   state, reward, done, _ = env.step(action)

   if done:
      if reward == 1: # reward is always in current players view, student gets reward -1 if opponent wins
         reward = -1
   env.change_player() # change back to student before returning
   return state, reward, done


def minimax(env, depth, alpha, beta, maximize, player = True, player_disc = 1) -> (int, int):
   if depth == 0 or len(env.available_moves()) == 0:
      score = utility(env.board, discType=player_disc)
      return score, -1
   else: # Continue with algorithm
      state = env.board
      avmoves = list(env.available_moves())
      np.random.shuffle(avmoves)
      best_action = np.random.choice(avmoves)
      if maximize:
         value = float('-inf')
         for a in avmoves:
            successor, reward, done, _ = env.step(a)
            if reward == 1: return 10000000000, a
            next_value, _ = minimax(env, depth - 1, alpha, beta, False, player_disc)
            state = env.reset(state)
            if next_value > value:
               value = next_value
               best_action = a
            alpha = max(alpha, value)
            if alpha >= beta:
               break
         return value, best_action

      else: # Minimize (opponent)
         env.change_player() # Sets current player to -1 (opponent)
         value = float('inf')
         for a in avmoves:
            successor, reward, done, _ = env.step(a)
            if reward == 1: return -10000000000, a
            next_value, _ = minimax(env, depth - 1, alpha, beta, True, player_disc)
            state = env.reset(state)
            env.change_player() # Reset sets current player to 1, so we have to revert back to -1
            if next_value < value:
               value = next_value
               best_action = a
            beta = min(beta, value)
            if alpha >= beta:
               break
         return value, best_action


def student_move(env, depth_lim):
   """
   TODO: Implement your min-max alpha-beta pruning algorithm here.
   Give it whatever input arguments you think are necessary
   (and change where it is called).
   The function should return a move from 0-6
   """
   start_time = time.time()
   avmoves = env.available_moves()
   if not avmoves:
      env.change_player()
      return -1

   state = env.board

   best_action = -1
   if 3 in avmoves and env.board[5][3] == 0:
      best_action = 3
   else:
      # Returns action from possible actions in state that maximizes the min_value(result(a, state))
      result, best_action = minimax(env, depth_lim, float('-inf'), float('inf'), True)

   env.reset(state)
   
   print(f'Time taken for move {time.time() - start_time}')
   return best_action


def play_game(vs_server = False, depth_lim = 5, random_bot = False):
   """
   The reward for a game is as follows. You get a
   botaction = random.choice(list(avmoves)) reward from the
   server after each move, but it is 0 while the game is running
   loss = -1
   win = +1
   draw = +0.5
   error = -10 (you get this if you try to play in a full column)
   Currently the player always makes the first move
   """

   # default state
   state = np.zeros((6, 7), dtype=int)

   # setup new game
   if vs_server:
      # Start a new game
      res = call_server(-1) # -1 signals the system to start a new game. any running game is counted as a loss

      # This should tell you if you or the bot starts
      print(res.json()['msg'])
      botmove = res.json()['botmove']
      state = np.array(res.json()['state'])
   else:
      # reset game to starting state
      env.reset(board=None)
      # determine first player
      student_gets_move = random.choice([True, False])
      if student_gets_move:
         print('You start!')
         print()
      else:
         print('Bot starts!')
         print()

   # Print current gamestate
   print("Current state (1 are student discs, -1 are servers, 0 is empty): ")
   print(state)
   print()

   done = False
   while not done:
      # Select your move
      stmove = student_move(env, depth_lim) # TODO: change input here
      # make both student and bot/server moves
      if vs_server:
         # Send your move to server and get response
         res = call_server(stmove)
         print(res.json()['msg'])

         # Extract response values
         result = res.json()['result']
         botmove = res.json()['botmove']
         state = np.array(res.json()['state'])

         env.reset(state)
         # env.step(botmove)
         # env.change_player()
      else:
         if student_gets_move:
            # Execute your move
            avmoves = env.available_moves()
            if stmove not in avmoves:
               print("You tied to make an illegal move! Games ends.")
               break
            state, result, done, _ = env.step(stmove)

         student_gets_move = True # student only skips move first turn if bot starts

         # print or render state here if you like

         # select and make a move for the opponent, returned reward from students view
         if not done:
            state, result, done = opponents_move(env, depth_lim, random_bot)

      # Check if the game is over
      if result != 0:
         done = True
         if not vs_server:
            print("Game over. ", end="")
         if result == 1:
            print("You won!")
            return True
         elif result == 0.5:
            print("It's a draw!")
            return True
         elif result == -1:
            print("You lost!")
         elif result == -10:
            print("You made an illegal move and have lost!")
         else:
            print("Unexpected result result={}".format(result))
         if not vs_server:
            print("Final state (1 are student discs, -1 are servers, 0 is empty): ")
         return False
      else:
         print("Current state (1 are student discs, -1 are servers, 0 is empty): ")

      # Print current gamestate
      print(state)
      print()

def play_many_games(vs_server, depth_lim, num_games, random_bot):
   results = np.arange(num_games)
   for i in np.arange(num_games):
      results[i] = play_game(vs_server, depth_lim, random_bot)
   true_or_draw = [x for x in results if x]
   print(f'Fraction of games won or tied: {len(true_or_draw)} / {len(results)}')

def main(argv):
   vs_server = True
   depth_limit = 5
   num_games = 1
   random_bot = "manual"
   
   try:
      opts, args = getopt.getopt(argv[1:],"b:d:g:",["bot=", "depth=", "games="])
   except getopt.GetoptError:
      raise SystemExit(f'\nExecute script by entering: python {sys.argv[0]} [-b,--bot <"random","ai","manual">] [OPTIONS]\n\nOPTIONS:\n\t[-d,--depth <int>]\n\t[-g,--games <int>]')
      sys.exit(2)
   for opt, arg in opts:
      if opt in ('-b', '--bot'):
         if arg.lower() not in ('random', 'ai', 'manual'):
            print("Wrong type entered, should be %s".format('r/random'))
            sys.exit(2)
         else:
            vs_server = False
            if arg.lower() == 'ai': random_bot = 'ai'
            elif arg.lower() == 'manual': random_bot = 'manual'
            else: random_bot = 'random'
      elif opt in ('-d', '--depth'):
         if int(arg): depth_limit = int(arg)
         else:
            print('Incorrect argument value type!')
            sys.exit(2)
      elif opt in ('-g', '--games'):
         if int(arg): num_games = int(arg)
         else:
            print('Incorrect argument value type!')
            sys.exit(2)

   if num_games > 1: play_many_games(vs_server, depth_limit, num_games, random_bot)
   else: play_game(vs_server, depth_limit, random_bot)

if __name__ == "__main__":
    main(sys.argv)
