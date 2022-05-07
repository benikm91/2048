import pygame
from pygame.locals import *
from game2048.r_learning import *


# I took the core of this game visualisation code from someone's github repo several weeks ago
# when i started doing this project. I love the simplicity and psychedelic colors.
# It's a shame i forgot where exactly i got it from. If you are the creator - write me
# and i will give you the credit in the readme file.

class Show:

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (244, 67, 54)
    PINK = (234, 30, 99)
    PURPLE = (156, 39, 176)
    DEEP_PURPLE = (103, 58, 183)
    BLUE = (33, 150, 243)
    TEAL = (0, 150, 136)
    L_GREEN = (139, 195, 74)
    GREEN = (60, 175, 80)
    ORANGE = (255, 152, 0)
    DEEP_ORANGE = (255, 87, 34)
    BROWN = (121, 85, 72)
    SCARLET = (255, 0, 30)
    PINK_Y = (220, 120, 120)

    colour = [BLACK, RED, PINK, PURPLE, DEEP_PURPLE, BLUE, TEAL, L_GREEN, GREEN, ORANGE,
              DEEP_ORANGE, BROWN, SCARLET, PINK_Y, PURPLE, DEEP_PURPLE, RED]

    def __init__(self):
        self.game = Game()
        pygame.init()
        pygame.display.set_caption("2048")
        self.board = pygame.display.set_mode((600, 700), 0, 32)
        self.font = pygame.font.SysFont('monospace', 25)

    def display(self, replay_move=None, over=False):
        if replay_move is not None:
            replay_move = Game.actions[replay_move]
        self.board.fill(Show.BLACK)
        if over:
            message = self.font.render(
                f'Over! score {self.game.score}, moves {self.game.odometer}', True, Show.WHITE)
        elif replay_move is None:
            message = self.font.render(
                f'score = {self.game.score} after {self.game.odometer} moves', True, Show.WHITE)
        else:
            message = self.font.render(
                f'score {self.game.score}, moves {self.game.odometer}, now = {replay_move}', True, Show.WHITE)
        self.board.blit(message, (10, 20))
        for i in range(4):
            for j in range(4):
                v = self.game.row[j, i]
                v_disp = 1 << v if v else 0
                pygame.draw.rect(self.board, Show.colour[v], (i * 150 + 2, j * 150 + 100 + 2, 146, 146))
                number = self.font.render(str(v_disp), True, Show.WHITE)
                if v_disp < 10:
                    offset = 10
                elif v_disp < 100:
                    offset = 15
                elif v_disp < 1000:
                    offset = 20
                elif v_disp < 10000:
                    offset = 25
                else:
                    offset = 30
                if v:
                    self.board.blit(number, (i * 150 + 75 - offset, j * 150 + 160))

    # play yourself

    def play(self):
        over = False
        while True:
            self.display(over=over)
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if self.game.game_over(self.game.row):
                    over = True
                elif event.type == KEYDOWN:
                    k, direction = event.key, -1
                    if k == pygame.K_LEFT:
                        direction = 0
                    elif k == pygame.K_UP:
                        direction = 1
                    elif k == pygame.K_RIGHT:
                        direction = 2
                    elif k == pygame.K_DOWN:
                        direction = 3
                    if direction >= 0:
                        change = self.game.make_move(direction)
                        if change:
                            self.game.new_tile()
                if event.type == KEYDOWN and event.key == pygame.K_r:
                    self.game = Game()
                    self.play()
            pygame.display.update()

    # replay a game from it's game.history

    def replay(self, game_to_show: Game, speed=1000):
        i = 0
        self.game = Game(row=game_to_show.starting_position)
        while i <= game_to_show.odometer:
            if i == game_to_show.odometer:
                self.display(over=True, replay_move=None)
            else:
                move = game_to_show.moves[i]
                tile, position = game_to_show.tiles[i]
                self.display(over=False, replay_move=move)
                self.game.make_move(move)
                self.game.row[position] = tile
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    return
            i += 1
            pygame.display.update()
            pygame.time.wait(speed)

    # watch an algorithm (estimator parameter) play on-line

    def watch(self, estimator, depth=0, width=1, ample=6, game_init=None, speed=500):
        game = game_init or Game()
        for state, move in game.generate_run(estimator=estimator, depth=depth, width=width, ample=ample):
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    return
            self.game = state
            self.display(replay_move=move, over=False)
            pygame.display.update()
            pygame.time.wait(speed)
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    return
            self.display(over=True)
            pygame.display.update()
        pass


if __name__ == "__main__":

    a = Q_agent.load_agent('agent_4.pkl')
    print(a.step, a.next_decay, a.alpha, a.low_alpha_limit, a.decay)
    a.next_decay = 34800
    a.save_agent()
    sys.exit()

    # The agent actually plays a game to 2048 in about 1 second. I set the speed of replays at 5 moves/sec,
    # change the speed parameter in ms below if you like

    print('option 0 = play yourself. Not sure why anybody would want it on a PC, but there os an option :)')
    print('option 1 = replay the best game in the best_game.npy file')
    print('option 2 = load the trained agent from best_agent.npy file. Play 100 games, replay the best')
    print('any other input - load the trained agent from best_agent.npy file and see it play on-line')

    option = int(input())
    if option == 0:
        Show().play()
    elif option == 1:
        game = Game.load_game("best_game.npy")
        Show().replay(game, speed=25)
    elif option == 2:
        agent = Q_agent.load_agent("best_agent.npy")
        est = agent.evaluate
        results = Game.trial(estimator=est, num=100)
        Show().replay(results[0], speed=200)
    else:
        agent = Q_agent.load_agent("best_agent.npy")
        est = agent.evaluate
        Show().watch(estimator=est, speed=20)
