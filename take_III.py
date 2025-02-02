from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from typing import Callable, Dict, Generator, List, Tuple


W_P, W_R, W_N, W_B, W_Q, W_K = "PRNBQK"
B_P, B_R, B_N, B_B, B_Q, B_K = "prnbqk"
EMPTY = " "

ROWS = '12345678'
COLS = 'abcdefgh'


Position = str # ex. 'e2'
PieceStr = str # ex. 'P'
Move = str # ex. 'e2e4' (UCI)


default_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'


@dataclass
class Game:
    board: Dict[Position, PieceStr]
    turn: str
    castling_rights: str
    en_passant: str
    half_moves: int
    move_number: int

    @staticmethod
    def from_fen(fen: str = default_fen) -> 'Game':
        (
            positions,
            turn,
            castling_rights,
            en_passant,
            half_moves,
            move_number
        ) = fen.split(' ')

        def _split_and_expand(line: str) -> Generator[str, None, None]:
            "helper generator function - from the compressed fen's board representation to full 8 characters row"
            for char in line:
                if char.isdigit():
                    for _ in range(int(char)):
                        yield EMPTY
                else:
                    yield char

        board = {
            f'{col_n}{row_n}': piece_str
            for row_n, row in zip(reversed(ROWS), positions.split('/'))
            for col_n, piece_str in zip(COLS, _split_and_expand(row))
        }

        return Game(
            board=board,
            turn=turn,
            castling_rights=castling_rights,
            en_passant=en_passant,
            half_moves=int(half_moves),
            move_number=int(move_number)
        )

    def _display_board(self, display_what: Callable[[Position], str]) -> None:
        for row in reversed(ROWS):
            for col in COLS:
                position = f'{col}{row}'
                print(display_what(position), end='')
            print()

    def _display_board_simple(self) -> None:
        self._display_board(lambda position: (
            self.board[position] if (self.board[position] != ' ') else ('*' if position == self.en_passant else '.')
        ))

    def display(self) -> None:
        print()
        self._display_board_simple()
        print()
        print(f'turn={self.turn}')
        print(f'castling_rights={self.castling_rights}')
        print(f'en_passant={self.en_passant}')
        print(f'half_moves={self.half_moves}')
        print(f'move_number={self.move_number}')
        print()

    def all_pieces(self, player: str) -> Generator[Tuple[PieceStr, Position], None, None]:
        for position, piece_str in self.board.items():
            if piece_str == EMPTY:
                continue
            if (piece_str.isupper() and player == 'b') or (piece_str.islower() and player == 'w'):
                continue
            yield piece_str, position

    def valid(self) -> bool:
        return True # TODO:


@dataclass(repr=False)
class Square():
    "helper class to address a single square on the board"
    position: Position
    _row: int = field(init=False)
    _col: int = field(init=False)

    def __post_init__(self):
        col, row = self.position
        self._col = ord(col) - ord(COLS[0]) # 0 based
        self._row = ord(row) - ord(ROWS[0]) # 0 based

    @property
    def row(self) -> str:
        return self.position[1]

    @property
    def col(self) -> str:
        return self.position[0]

    def add(self, rows: int, cols: int) -> Position | None:
        row = self._row + rows
        col = self._col + cols
        if row < 0 or row > 7:
            return None
        if col < 0 or col > 7:
            return None
        return f'{chr(col + ord("a"))}{row + 1}'

    def __repr__(self):
        return self.position


class Piece(ABC):
    def __init__(self, piece_str: PieceStr, position: Position):
        self._piece_str = piece_str
        self._square = Square(position)

    @abstractmethod
    def possible_moves(self) -> Generator[Tuple[Move, Game],None,None]:
        pass

    def same_color(self, other_piece_str: PieceStr) -> bool:
        return self._piece_str.isupper() == other_piece_str.isupper()

    def attempt_move(self, game: Game, move_square_str: Position) -> Tuple[Move, Game] | None:
        next_game_state = deepcopy(game)
        next_game_state.board[str(self._square)] = EMPTY
        next_game_state.board[move_square_str] = self._piece_str
        if not next_game_state.valid():
            return None
        return f'{self._square}{move_square_str}', next_game_state


class Pawn(Piece):
    def __init__(self, piece_str: PieceStr, position: Position):
        super().__init__(piece_str=piece_str, position=position)
        assert piece_str == W_P or piece_str == B_P
        self._direction = +1 if self._piece_str == W_P else -1
        starting_row = '2' if self._piece_str == W_P else '7'
        pre_promotion_row = '7' if self._piece_str == W_P else '2'
        col, row = self._square.col, self._square.row
        self.pre_promotion = pre_promotion_row == row
        self._theoretical_moves = [self._square.add(rows=1 * self._direction, cols=0)]
        if row == starting_row:
            self._theoretical_moves.append(self._square.add(rows=2 * self._direction, cols=0))
        self._theoretical_captures = []
        if col > COLS[0]:
            self._theoretical_captures.append(self._square.add(rows=1 * self._direction, cols=-1))
        if col < COLS[-1]:
            self._theoretical_captures.append(self._square.add(rows=1 * self._direction, cols=+1))

    def possible_moves(self, game: Game) -> Generator[Tuple[Move, Game],None,None]:
        two_steps = False
        for move_square_str in self._theoretical_moves:
            piece_there_str = game.board[move_square_str]
            if piece_there_str == EMPTY:
                ret = self.attempt_move(game, move_square_str)
                if not ret:
                    continue
                move, next_game_state = ret
                if self._piece_str == B_N:
                    assert game.turn == 'b'
                    next_game_state.move_number += 1
                    next_game_state.turn = 'w'
                else:
                    assert game.turn == 'w'
                    next_game_state.turn = 'b'
                next_game_state.en_passant = self._square.add(1 * self._direction, 0) if two_steps else '-'
                next_game_state.half_moves = 0
                yield move, next_game_state
                two_steps = True
            else:
                break
        for capture_square_str in self._theoretical_captures:
            piece_there_str = game.board[capture_square_str]
            if piece_there_str == EMPTY:
                if game.en_passant != capture_square_str:
                    continue
            elif self.same_color(piece_there_str):
                continue
            ret = self.attempt_move(game, move_square_str)
            if not ret:
                continue
            move, next_game_state = ret
            if self._piece_str == B_P:
                assert game.turn == 'b'
                next_game_state.move_number += 1
                next_game_state.turn = 'w'
            else:
                assert game.turn == 'w'
                next_game_state.turn = 'b'
            next_game_state.half_moves = 0
            next_game_state.en_passant = '-'
            yield move, next_game_state

        # TODO: promotion


class Knight(Piece):
    def __init__(self, piece_str: PieceStr, position: Position):
        super().__init__(piece_str=piece_str, position=position)
        assert piece_str == W_N or piece_str == B_N
        self._theoretical_moves = [
            position
            for add_row, add_col in zip([1, 2], [2, 1])
            for sign_row, sign_col in product([-1, 1], [-1, 1])
            if (position := self._square.add(rows=sign_row * add_row, cols=sign_col * add_col)) is not None
        ]

    def possible_moves(self, game: Game) -> Generator[Tuple[Move, Game],None,None]:
        for move_square_str in self._theoretical_moves:
            piece_there_str = game.board[move_square_str]
            if piece_there_str != EMPTY and self.same_color(piece_there_str):
                continue
            ret = self.attempt_move(game, move_square_str)
            if not ret:
                continue
            move, next_game_state = ret
            capture = piece_there_str != EMPTY
            if self._piece_str == B_N:
                assert game.turn == 'b'
                next_game_state.move_number += 1
                # next_game_state.castling_rights = next_game_state.castling_rights.remove('kq')
                next_game_state.turn = 'w'
            else:
                assert game.turn == 'w'
                # next_game_state.castling_rights = next_game_state.castling_rights.remove('KQ')
                next_game_state.turn = 'b'
            next_game_state.half_moves = 0 if capture else next_game_state.half_moves + 1
            next_game_state.en_passant = '-'
            yield move, next_game_state


class RayBased(Piece, ABC):
    def __init__(self, piece_str: PieceStr, position: Position):
        super().__init__(piece_str=piece_str, position=position)
        self._rays = [
            [target_square_str for target_square_str in ray if target_square_str]
            # if a target_square_str is "outside of the board" we shall have it here as None
            for ray in self.get_rays()
        ]

    @abstractmethod
    def get_rays(self) -> List[List[str]]:
        pass

    def possible_moves(self, game: Game) -> Generator[Tuple[Move, Game],None,None]:
        for ray in self._rays:
            for target_square_str in ray:
                piece_there_str = game.board[target_square_str]
                if piece_there_str != EMPTY and self.same_color(piece_there_str):
                    break # this ray reached a piece
                ret = self.attempt_move(game, target_square_str)
                if not ret:
                    continue
                move, next_game_state = ret
                capture = piece_there_str != EMPTY
                if self.same_color(B_K):
                    assert game.turn == 'b'
                    next_game_state.move_number += 1
                    # next_game_state.castling_rights = next_game_state.castling_rights.replace('k', '')
                    # next_game_state.castling_rights = next_game_state.castling_rights.replace('q', '')
                    next_game_state.turn = 'w'
                else:
                    assert game.turn == 'w'
                    # next_game_state.castling_rights = next_game_state.castling_rights.replace('K', '')
                    # next_game_state.castling_rights = next_game_state.castling_rights.replace('Q', '')
                    next_game_state.turn = 'b'
                # if next_game_state.castling_rights == '':
                #     next_game_state.castling_rights = '-' 
                next_game_state.half_moves = 0 if capture else next_game_state.half_moves + 1
                next_game_state.en_passant = '-'
                yield move, next_game_state


class RookRays():
    "mixin"
    def _get_rook_rays(self) -> List[List[str]]:
        return [
            [self._square.add(rows=0, cols=cols) for cols in range(1, 8)],
            [self._square.add(rows=0, cols=cols) for cols in range(-1, -8, -1)],
            [self._square.add(rows=rows, cols=0) for rows in range(1, 8)],
            [self._square.add(rows=rows, cols=0) for rows in range(-1, -8, -1)],
        ]


class Rook(RayBased, RookRays):
    def __init__(self, piece_str: PieceStr, position: Position):
        super().__init__(piece_str=piece_str, position=position)
        assert piece_str == W_R or piece_str == B_R

    def get_rays(self) -> List[List[str]]:
        return self._get_rook_rays()


class BishopRays():
    "mixin"
    def _get_bishop_rays(self) -> List[List[str]]:
        return [
            [self._square.add(r, c) for r, c in zip(range(1, 8), range(1, 8))],
            [self._square.add(r, c) for r, c in zip(range(-1, -8, -1), range(1, 8))],
            [self._square.add(r, c) for r, c in zip(range(1, 8), range(-1, -8, -1))],
            [self._square.add(r, c) for r, c in zip(range(-1, -8, -1), range(-1, -8, -1))],
        ]


class Bishop(RayBased, BishopRays):
    def __init__(self, piece_str: PieceStr, position: Position):
        super().__init__(piece_str=piece_str, position=position)
        assert piece_str == W_B or piece_str == B_B

    def get_rays(self) -> List[List[str]]:
        return self._get_bishop_rays()


class Queen(RayBased, RookRays, BishopRays):
    def __init__(self, piece_str: PieceStr, position: Position):
        super().__init__(piece_str=piece_str, position=position)
        assert piece_str == W_Q or piece_str == B_Q

    def get_rays(self) -> List[List[str]]:
        return self._get_rook_rays() + self._get_bishop_rays()


class King(Piece):
    def __init__(self, piece_str: PieceStr, position: Position):
        super().__init__(piece_str=piece_str, position=position)
        assert piece_str == W_K or piece_str == B_K
        self._theoretical_moves = [
            position
            for add_rows, add_cols in product([-1, 0, 1], [-1, 0, 1]) if (
                (add_rows != 0 or add_cols != 0)
                and
                ((position := self._square.add(add_rows, add_cols)) is not None)
            )
        ]

    def possible_moves(self, game: Game) -> Generator[Tuple[Move, Game],None,None]:
        for move_square_str in self._theoretical_moves:
            piece_there_str = game.board[move_square_str]
            if piece_there_str != EMPTY and self.same_color(piece_there_str):
                continue
            ret = self.attempt_move(game, move_square_str)
            if not ret:
                continue
            move, next_game_state = ret
            capture = piece_there_str != EMPTY
            if self._piece_str == B_K:
                assert game.turn == 'b'
                next_game_state.move_number += 1
                next_game_state.castling_rights = next_game_state.castling_rights.replace('k', '')
                next_game_state.castling_rights = next_game_state.castling_rights.replace('q', '')
                next_game_state.turn = 'w'
            else:
                assert game.turn == 'w'
                next_game_state.castling_rights = next_game_state.castling_rights.replace('K', '')
                next_game_state.castling_rights = next_game_state.castling_rights.replace('Q', '')
                next_game_state.turn = 'b'
            if next_game_state.castling_rights == '':
                next_game_state.castling_rights = '-' 
            next_game_state.half_moves = 0 if capture else next_game_state.half_moves + 1
            next_game_state.en_passant = '-'
            yield move, next_game_state


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
            yield from piece.possible_moves(game)


def main():
    game = Game.from_fen()

    game.display()

    possible_moves = dict(GameEvaluation.possible_moves(game))

    for move, next_game_state in possible_moves.items():
        print()
        print(move)
        next_game_state.display()


if __name__ == "__main__":
    main()
