from agents import RandomPlayer, EGreedyPlayer


class TicTacToe:
    def __init__(self, player, opponent):
        """
        Initialises a Tic-Tac-Toe environment, by specifying the two players in the game.
        """
        self.board = [" " for x in range(9)]
        self.player = player
        self.opponent = opponent

        # init
        self.player_turn = True
        self.moves_left = 9
        return

    def reset(self):
        """
        Resets the environment to the initial state: an empty board
        """
        self.board = [" " for x in range(9)]
        self.moves_left = 9
        return self.board, False

    def step(self):
        """
        Performs one play for the player that holds its turn.
        This updates the board with the new move.
        """
        if self.player_turn:
            player = self.player
        else:
            player = self.opponent

        pos, reward = player.get_action(self.board)
        self.board[pos] = player.marker

        self.player_turn = not self.player_turn
        self.moves_left -= 1

        return self.board, reward, self.game_ended()

    def learn(self, n_games):
        """
        Plays `n_games` consecutively to let the player learn.
        Args:
            n_games (int): the number of games to learn from
        """
        self.player.train()
        for i in range(n_games):
            print("Playing game {}\t".format(i), end="\r")
            state, done = self.reset()
            while not done:
                state, reward, done = self.step()
        return self.player

    def play(self):
        """
        Tests the skills of each player in the game by having a single game, whilst learning is disabled.
        """
        self.player.eval()
        state, done = self.reset()
        while not done:
            state, reward, done = self.step()
            self.draw()
        return state, done

    def game_ended(self):
        """
        Checks if the game reached its conclusing, either because there are no moves left,
        or because one of the two player has won.
        """
        if self.moves_left <= 0:
            return True

        state = self.board
        # row
        for i in range(0, 9, 3):
            if (state[i] == state[i + 1] == state[i + 2]):
                return state[i] != " "
        # col
        for i in range(3):
            if (state[i] == state[i + 3] == state[i + 6]):
                return state[i] != " "
        # diag
        if (state[0] == state[4] == state[8]):
            return state[0] != " "
        # anti-diag
        if (state[2] == state[4] == state[6]):
            return state[2] != " "

        return False

    def draw(self):
        """
        Plots the current state of the board
        """
        for i in range(0, 9, 3):
            print(self.board[i:i + 3])
        print("State value:", self.player.get_value(self.board))
        print()


if __name__ == "__main__":
    opponent = RandomPlayer("O")
    player = EGreedyPlayer("X", init_value=0.5, e_greedy=0.3, step_size=0.5, decrement=0.8, decrement_each=10000)
    env = TicTacToe(opponent=opponent, player=player)

    env.learn(100000)

    env.play()
    print(player.step_size  )