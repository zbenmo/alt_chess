from collections import Counter
from functools import lru_cache
from typing import Generator, Tuple
from chess_game import (
    Position,
    Move,
    Game,
    PieceStr,
    W_K, W_B, W_N, W_P, W_Q, W_R,
    B_K, B_B, B_N, B_P, B_Q, B_R,
    EMPTY
)
from pieces import Piece, Pawn, Rook, Knight, Bishop, Queen, King


class GameEvaluation:

    @staticmethod
    @lru_cache
    def piece_for(piece_str: PieceStr, position: Position) -> Piece:
        match piece_str:
            case 'p' | 'P':
                return Pawn(piece_str, position=position)
            case 'r' | 'R':
                return Rook(piece_str, position=position)
            case 'n' | 'N':
                return Knight(piece_str, position=position)
            case 'b' | 'B':
                return Bishop(piece_str, position=position)
            case 'q' | 'Q':
                return Queen(piece_str, position=position)
            case 'k' | 'K':
                return King(piece_str, position=position)
            case _:
                assert False, f'{piece_str=}'
        pass

    @staticmethod
    def possible_moves(game: Game) -> Generator[Tuple[Move, Game],None,None]:
        for piece_str, position in game.all_pieces(player=game.turn):
            piece = GameEvaluation.piece_for(piece_str, position)
            yield from piece.possible_moves(game, GameEvaluation.is_checked)

    @staticmethod
    def is_checked(game: Game) -> bool:
        which_king = W_K if game.turn == 'w' else B_K
        kings_position = next(position for position, piece_str in game.board.items() if piece_str == which_king)
        other_player = 'b' if game.turn == 'w' else 'w'
        for piece_str, position in game.all_pieces(player=other_player):
            piece = GameEvaluation.piece_for(piece_str, position)
            if any(kings_position == position for position in piece.positions_threats(game)):
                return True
        return False

    @staticmethod
    def evaluate(game: Game, print_to_screen: bool=False):
        kings_positions = Counter()
        threats_white = Counter()
        threats_black = Counter()
        promotions = Counter()
        for position, piece_str in game.board.items():
            if piece_str == EMPTY:
                continue
            if piece_str == W_K:
                kings_positions[position] += 1
            elif piece_str == B_K:
                kings_positions[position] -= 1
            if piece_str == W_P and position[1] == '7':
                promotions[position] += 1
            elif piece_str == B_P and position[1] == '1':
                promotions[position] -= 1
            piece = GameEvaluation.piece_for(piece_str, position)
            if piece_str.isupper():
                for threat_position in piece.positions_threats(game):
                    threats_white[threat_position] += 1
            else:
                for threat_position in piece.positions_threats(game):
                    threats_black[threat_position] += 1
        if print_to_screen:
            print()
            print('threats white')
            print()
            game._display_board(lambda position: threats_white.get(position, '.'))
            print()
            print('threats black')
            print()
            game._display_board(lambda position: threats_black.get(position, '.'))
            print()
            print('kings positions')
            print()
            game._display_board(lambda position: kings_positions.get(position, '.'))
            print()
            print('promotions positions')
            print()
            game._display_board(lambda position: promotions.get(position, '.'))

    def take_move(game: Game, move: str) -> Game | None:
        from_position = move[:2]
        piece_str = game.board[from_position]
        assert piece_str != EMPTY
        is_upper: bool = piece_str.isupper()
        assert (game.turn == 'b') or is_upper, f'{game.turn=}, {piece_str=}'
        piece = GameEvaluation.piece_for(piece_str, from_position)
        for possible_move, next_game_state in piece.possible_moves(game, GameEvaluation.is_checked):
            if possible_move == move:
                return next_game_state

    def simple_heuristics(game: Game) -> Tuple[float, float]:
        counter = Counter(game.board.values())
        white = (
            counter[W_P] * 1.0
            + counter[W_B] * 3.0
            + counter[W_N] * 3.0
            + counter[W_R] * 5.0
            + counter[W_Q] * 9.0
        )
        black = (
            counter[B_P] * 1.0
            + counter[B_B] * 3.0
            + counter[B_N] * 3.0
            + counter[B_R] * 5.0
            + counter[B_Q] * 9.0
        )
        return white, black


# default_fen = 'rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1'
# default_fen = 'rnbqkbnr/ppppppPp/8/8/8/8/PPPPP1PP/RNBQKBNR w KQkq - 0 1'
# default_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQkq - 0 1'


def main():
    import timeit

    game = Game.from_fen() # pass a fen..

    game.display()

    possible_moves = dict(GameEvaluation.possible_moves(game))

    for move, next_game_state in possible_moves.items():
        print()
        print(move)
        next_game_state.display()

    GameEvaluation.evaluate(game, print_to_screen=True)

    # print(timeit.timeit(lambda: GameEvaluation.evaluate(game), number=10_000))

    move = "e2e4"

    next_game_state = GameEvaluation.take_move(game, move)

    next_game_state.display()

    print(GameEvaluation.simple_heuristics(next_game_state))


if __name__ == "__main__":
    main()
