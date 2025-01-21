from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Dict, Generator, List, Protocol
from events import Event


W_P, W_R, W_N, W_B, W_Q, W_K = "PRNBQK"
B_P, B_R, B_N, B_B, B_Q, B_K = "prnbqk"
EMPTY = " "

ROWS = '12345678'
COLS = 'abcdefgh'

# W_EN_PASSANT = 'E'
# B_EN_PASSANT = 'e'


@dataclass
class Square():
    "helper class to manage a single square on the board"
    position: str # ex. 'e2'
    piece_str: str = None
    _row: int = field(init=False)
    _col: int = field(init=False)
    white_can_move: int = field(init=False, default=0)
    black_can_move: int = field(init=False, default=0)
    white_can_capture: int = field(init=False, default=0)
    black_can_capture: int = field(init=False, default=0)
    white_can_promote: int = field(init=False, default=0)
    black_can_promote: int = field(init=False, default=0)
    changed: Event = field(init=False, default_factory=Event)

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

    def add(self, rows: int, cols: int) -> str | None:
        row = self._row + rows
        col = self._col + cols
        if row < 0 or row > 7:
            return None
        if col < 0 or col > 7:
            return None
        return f'{chr(col + ord("a"))}{row + 1}'

    def place(self, piece_str: str):
        self.piece_str = piece_str
        self.changed(square=self)

    def pick(self):
        self.piece_str = None
        self.changed(square=self)

    def indicate_can_move_to(self, piece_str: str, can_capture: bool = True, can_promote: bool = False):
        if piece_str.isupper():
            self.white_can_move += 1
            self.white_can_capture += int(can_capture)
            self.white_can_promote += int(can_promote)
        else:
            self.black_can_move += 1
            self.black_can_capture += int(can_capture)
            self.black_can_promote += int(can_promote)

    def __repr__(self):
        return self.position


class Board(Protocol):
    def get_square(position: str) -> Square:
        ... 


# @dataclass
# class Move:
#     target_square_str: str 


class Piece(ABC):
    def __init__(self, piece_str: str, position: str, board: Board):
        self._piece_str = piece_str
        self._position = position
        self._board = board
        self._moves = []

    @abstractmethod
    def complete_initialization(self):
        pass

    def add_move(self, target_sqaure_str: str) -> None:
        self._moves.append(target_sqaure_str)

    def same_color(self, other_piece_str: str) -> bool:
        return self._piece_str.isupper() == other_piece_str.isupper()

    def moves(self) -> Generator[str, None, None]:
        yield from [
            f'{self._position}{move}' for move in self._moves
        ]

    def __repr__(self):
        return f'{self._piece_str} at {self._position}'


class Pawn(Piece):
    def __init__(self, piece_str: str, position: str, board: Board):
        super().__init__(piece_str=piece_str, position=position, board=board)
        assert piece_str == W_P or piece_str == B_P
        square = self._board.get_square(self._position)
        self.direction = +1 if self._piece_str == W_P else -1
        self.starting_row = '2' if self._piece_str == W_P else '7'
        pre_promotion_row = '7' if self._piece_str == W_P else '2'
        self.col, self.row = square.col, square.row
        self.pre_promotion = pre_promotion_row == self.row
    
    def complete_initialization(self):
        square = self._board.get_square(self._position)
        potential_moves = [square.add(rows=1 * self.direction, cols=0)]
        if self.row == self.starting_row:
            potential_moves.append(square.add(rows=2 * self.direction, cols=0))
        previous_move_square = None
        for move_square_str in potential_moves:
            move_square = self._board.get_square(move_square_str)
            move_square.indicate_can_move_to(self._piece_str, can_capture=False, can_promote=self.pre_promotion)
            move_square.changed += (self._square_changed, )
            if move_square.piece_str is None:
                self.add_move(move_square_str)
                if previous_move_square is None:
                    previous_move_square = move_square
                else:
                    self._indicate_en_passant(previous_move_square)
            else:
                break
        potential_captures = []
        if self.col > COLS[0]:
            potential_captures.append(square.add(rows=1 * self.direction, cols=-1))
        if self.col < COLS[-1]:
            potential_captures.append(square.add(rows=1 * self.direction, cols=+1))
        for capture_square_str in potential_captures:
            capture_square = self._board.get_square(capture_square_str)
            capture_square.indicate_can_move_to(self._piece_str, can_capture=True, can_promote=self.pre_promotion)
            capture_square.changed += (self._square_changed, )
            if capture_square.piece_str is None:
                pass
            elif not self.same_color(capture_square.piece_str):
                self.add_move(capture_square_str)
            else:
                pass
            # TODO: en-passant
        square.changed += (self._square_changed, )

    def _indicate_en_passant(self, square: Square):
        square.changed += (self._square_changed, ) # ?

    def _square_changed(self, square: Square):
        print(f'{self._piece_str} _square_changed {square}')

    def moves(self) -> Generator[str, None, None]:
        if self.pre_promotion:
            promotions = [W_Q, W_R, W_B, W_N] if self._piece_str == W_P else [B_Q, B_R, B_B, B_N]
            for move, promotion in product(self._moves, promotions):
                yield f'{self._position}{move}{promotion}'
        else:
            yield from [
                f'{self._position}{move}' for move in self._moves
            ]


class Knight(Piece):
    def __init__(self, piece_str: str, position: str, board: Board):
        super().__init__(piece_str=piece_str, position=position, board=board)
        assert piece_str == W_N or piece_str == B_N

    def complete_initialization(self):
        square = self._board.get_square(self._position)
        for add_row, add_col in zip([1, 2], [2, 1]):
            for sign_row, sign_col in product([-1, 1], [-1, 1]):
                target_square_str = square.add(rows=sign_row * add_row, cols=sign_col * add_col)
                if target_square_str is None:
                    continue
                target_square = self._board.get_square(target_square_str)
                target_square.indicate_can_move_to(self._piece_str)
                if target_square.piece_str is None:
                    self.add_move(target_square_str)
                elif not self.same_color(target_square.piece_str):
                    self.add_move(target_square_str)
                #TODO: events


class RayBased(Piece, ABC):
    def __init__(self, piece_str: str, position: str, board: Board):
        super().__init__(piece_str=piece_str, position=position, board=board)

    def complete_initialization(self):
        for ray in self.get_rays():
            for target_square_str in ray:
                if target_square_str is None:
                    break
                target_square = self._board.get_square(target_square_str)
                target_square.indicate_can_move_to(self._piece_str)
                if target_square.piece_str is None:
                    self.add_move(target_square_str)
                else:
                    if not self.same_color(target_square.piece_str):
                        self.add_move(target_square_str) # capture
                    break # this ray reached a piece
                    #TODO: events

    @abstractmethod
    def get_rays(self) -> List[List[str]]:
        pass


class RookRays():
    "mixin"
    def _get_rook_rays(self) -> List[List[str]]:
        square = self._board.get_square(self._position)
        return [
            [square.add(rows=0, cols=cols) for cols in range(1, 8)],
            [square.add(rows=0, cols=cols) for cols in range(-1, -8, -1)],
            [square.add(rows=rows, cols=0) for rows in range(1, 8)],
            [square.add(rows=rows, cols=0) for rows in range(-1, -8, -1)],
        ]


class Rook(RayBased, RookRays):
    def __init__(self, piece_str: str, position: str, board: Board):
        super().__init__(piece_str=piece_str, position=position, board=board)
        assert piece_str == W_R or piece_str == B_R

    def get_rays(self) -> List[List[str]]:
        return self._get_rook_rays()


class BishopRays():
    "mixin"
    def _get_bishop_rays(self) -> List[List[str]]:
        square = self._board.get_square(self._position)
        return [
            [square.add(r, c) for r, c in zip(range(1, 8), range(1, 8))],
            [square.add(r, c) for r, c in zip(range(-1, -8, -1), range(1, 8))],
            [square.add(r, c) for r, c in zip(range(1, 8), range(-1, -8, -1))],
            [square.add(r, c) for r, c in zip(range(-1, -8, -1), range(-1, -8, -1))],
        ]


class Bishop(RayBased, BishopRays):
    def __init__(self, piece_str: str, position: str, board: Board):
        super().__init__(piece_str=piece_str, position=position, board=board)
        assert piece_str == W_B or piece_str == B_B

    def get_rays(self) -> List[List[str]]:
        return self._get_bishop_rays()


class Queen(RayBased, RookRays, BishopRays):
    def __init__(self, piece_str: str, position: str, board: Board):
        super().__init__(piece_str=piece_str, position=position, board=board)
        assert piece_str == W_Q or piece_str == B_Q

    def get_rays(self) -> List[List[str]]:
        return self._get_rook_rays() + self._get_bishop_rays()


class King(Piece):
    def __init__(self, piece_str: str, position: str, board: Board):
        super().__init__(piece_str=piece_str, position=position, board=board)
        assert piece_str == W_K or piece_str == B_K

    def complete_initialization(self):
        square = self._board.get_square(self._position)
        for add_rows, add_cols in product([-1, 0, 1], [-1, 0, 1]):
            if add_rows == 0 and add_cols == 0:
                continue
            target_square_str = square.add(add_rows, add_cols)
            if target_square_str is None:
                continue
            target_square = self._board.get_square(target_square_str)
            target_square.indicate_can_move_to(self._piece_str) # TODO: only if not being thretened
            if target_square.piece_str is None:
                self.add_move(target_square_str)
            else:
                if not self.same_color(target_square.piece_str):
                    self.add_move(target_square_str) # capture
            #TODO: events


default_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'


class Game():
    def __init__(self):
        self.board: Dict[str, Square] = {}
        for row in ROWS:
            for col in COLS:
                position = f'{col}{row}'
                self.board[position] = Square(position=position)
        self.pieces = []
        self.turn = None
        self.castling_rights = None
        self.en_passant = None
        self.half_moves = None
        self.move_number = None

    @staticmethod
    def from_fen(fen: str = default_fen) -> 'Game':
        game = Game()

        (
            positions,
            turn,
            castling_rights,
            en_passant,
            half_moves,
            move_number
        ) = fen.split(' ')

        game.turn = turn
        game.castling_rights = castling_rights
        game.en_passant = en_passant
        game.half_moves = half_moves
        game.move_number = move_number

        def split_and_expand(line: str) -> Generator[str, None, None]:
            "helper generator function - from the compressed fen's board representation to full 8 characters row"
            for char in line:
                if char.isdigit():
                    for _ in range(int(char)):
                        yield EMPTY
                else:
                    yield char

        piece_placement = [
            list(split_and_expand(line))
            for line in reversed(positions.split('/'))
        ]

        for row_ind, row in zip(ROWS, piece_placement):
            for col_ind, piece_str in zip(COLS, row):
                if piece_str == EMPTY:
                    continue
                position = f'{col_ind}{row_ind}'
                game.place(position=position, piece_str=piece_str)

        game.complete_initialization()

        return game

    def _display_board(self, display_what: Callable[[Square], str]) -> None:
        for row in reversed(ROWS):
            for col in COLS:
                position = f'{col}{row}'
                square = self.get_square(position)
                print(display_what(square), end='')
            print()

    def display(self) -> None:
        print()
        self._display_board(lambda square: square.piece_str or '.')
        print()

        print(f'turn={self.turn}')
        print(f'castling_rights={self.castling_rights}')
        print(f'en_passant={self.en_passant}')
        print(f'half_moves={self.half_moves}')
        print(f'move_number={self.move_number}')
        print()

        print('white can potentially move:')
        print()
        self._display_board(lambda square: square.white_can_move)
        print()
        print('black can potentially move:')
        print()
        self._display_board(lambda square: square.black_can_move)
        print()

        print('white can potentially capture:')
        print()
        self._display_board(lambda square: square.white_can_capture)
        print()
        print('black can potentially capture:')
        print()
        self._display_board(lambda square: square.black_can_capture)
        print()

        print('white can potentially promote:')
        print()
        self._display_board(lambda square: square.white_can_promote)
        print()
        print('black can potentially promote:')
        print()
        self._display_board(lambda square: square.black_can_promote)
        print()

        print('kings position')
        print()
        self._display_board(lambda square: square.piece_str if square.piece_str in [W_K, B_K] else '.')
        print()

    @staticmethod
    def piece_for(piece_str: str, position: str, board: Board) -> Piece:
        match piece_str:
            case 'p' | 'P':
                return Pawn(piece_str, position=position, board=board)
            case 'r' | 'R':
                return Rook(piece_str, position=position, board=board)
            case 'n' | 'N':
                return Knight(piece_str, position=position, board=board)
            case 'b' | 'B':
                return Bishop(piece_str, position=position, board=board)
            case 'q' | 'Q':
                return Queen(piece_str, position=position, board=board)
            case 'k' | 'K':
                return King(piece_str, position=position, board=board)
            case _:
                assert False, f'{piece_str=}'

    def place(self, position: str, piece_str: str):
        piece = Game.piece_for(piece_str=piece_str, position=position, board=self)
        self.pieces.append(piece)
        square = self.get_square(position)
        square.piece_str = piece_str

    def get_square(self, position: str) -> Square:
        return self.board[position]

    def complete_initialization(self) -> 'Game':
        for piece in self.pieces:
            piece.complete_initialization()
        return self


def main():
    game = Game.from_fen()

    # pawn = Pawn('P')
    # game.place('e2', pawn)
    # game.get_square('e2').pick()

    game.display()

    for piece in game.pieces:
        print(f'{piece=}, moves={list(piece.moves())}')


if __name__ == "__main__":
    main()
