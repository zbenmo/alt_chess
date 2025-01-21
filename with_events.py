from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Generator, List, Protocol
from events import Event


W_P, W_R, W_N, W_B, W_Q, W_K = "PRNBQK"
B_P, B_R, B_N, B_B, B_Q, B_K = "prnbqk"
EMPTY = " "


@dataclass
class Square():
    position: str
    piece_str: str = None
    _row: int = field(init=False)
    _col: int = field(init=False)
    changed: Event = field(init=False, default_factory=Event)
    can_capture: int = field(init=False, default=0)
    can_promote: int = field(init=False, default=0)

    def __post_init__(self):
        col, row = self.position
        self._col = ord(col) - ord('a') # 0 based
        self._row = int(row) - 1 # 0 based

    @property
    def row(self) -> str:
        return self.position[1]

    @property
    def col(self) -> str:
        return self.position[0]

    def add(self, rows: int, cols: int) -> str:
        row = self._row + rows
        col = self._col + cols
        return f'{chr(col + ord("a"))}{row + 1}'

    def place(self, piece_str: str):
        self.piece_str = piece_str
        self.square_changed(square=self)

    def pick(self):
        self.piece_str = None
        self.changed(square=self)

    def indicate_can_move_to(self, can_capture: bool = True, can_promote: bool = False):
        self.can_capture += 1
        self.can_promote += 1

    def __repr__(self):
        return self.position


class Board(Protocol):
    def get_square(position: str) -> Square:
        ... 


class Piece(ABC):
    def __init__(self, piece_str: str, position: str, board: Board):
        self._piece_str = piece_str
        self._position = position
        self._board = board
        self._moves = []

    @abstractmethod
    def complete_initialization(self):
        pass


class Pawn(Piece):
    def __init__(self, piece_str: str, position: str, board: Board):
        super().__init__(piece_str=piece_str, position=position, board=board)
        assert piece_str == W_P or piece_str == B_P
    
    def complete_initialization(self):
        square = self._board.get_square(self._position)
        direction = +1 if self._piece_str == W_P else -1
        starting_row = '2' if self._piece_str == W_P else '7'
        pre_promotion_row = '7' if self._piece_str == W_P else '2'
        col, row = square.col, square.row
        pre_promotion = pre_promotion_row == row
        potential_moves = [square.add(rows=1 * direction, cols=0)]
        if row == starting_row:
            potential_moves.append(square.add(rows=2 * direction, cols=0))
        for move_square_str in potential_moves:
            move_square = self._board.get_square(move_square_str)
            move_square.indicate_can_move_to(can_capture=False, can_promote=pre_promotion)
            # if abs(move_square.row() - row) == 2:

            move_square.changed += (self._square_changed, )
            if move_square.piece_str is None:
                self._moves.append(move_square_str)
            else:
                break
            # TODO: the second move is also associated with en-passant
            # TODO: promotion?
            # TODO: indicate to the target square...
        potential_captures = []
        if col > 'a':
            potential_captures.append(square.add(rows=1 * direction, cols=-1))
        if col < 'h':
            potential_captures.append(square.add(rows=1 * direction, cols=+1))
        for capture_square_str in potential_captures:
            capture_square = self._board.get_square(capture_square_str)
            capture_square.indicate_can_move_to(can_capture=True, can_promote=pre_promotion)
            capture_square.changed += (self._square_changed, )
            if capture_square.piece_str is None:
                pass
            elif capture_square.piece_str.isupper() == self._piece_str.isupper():
                self._moves.append(capture_square)
            else:
                pass
            # TODO: en-passant
            # TODO: promotion?
        square.changed += (self._square_changed, )

    def _square_changed(self, square: Square):
        print(f'{self._piece_str} _square_changed {square}')


class Knight(Piece):
    def __init__(self, piece_str: str, position: str, board: Board):
        super().__init__(piece_str=piece_str, position=position, board=board)
        assert piece_str == W_N or piece_str == B_N

    def complete_initialization(self):
        pass


class Rook(Piece):
    def __init__(self, piece_str: str, position: str, board: Board):
        super().__init__(piece_str=piece_str, position=position, board=board)
        assert piece_str == W_R or piece_str == B_R

    def complete_initialization(self):
        pass


class Bishop(Piece):
    def __init__(self, piece_str: str, position: str, board: Board):
        super().__init__(piece_str=piece_str, position=position, board=board)
        assert piece_str == W_B or piece_str == B_B

    def complete_initialization(self):
        pass


class Queen(Piece):
    def __init__(self, piece_str: str, position: str, board: Board):
        super().__init__(piece_str=piece_str, position=position, board=board)
        assert piece_str == W_Q or piece_str == B_Q

    def complete_initialization(self):
        pass


class King(Piece):
    def __init__(self, piece_str: str, position: str, board: Board):
        super().__init__(piece_str=piece_str, position=position, board=board)
        assert piece_str == W_K or piece_str == B_K

    def complete_initialization(self):
        pass


default_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'


class Game():
    def __init__(self):
        self.board: Dict[str, Square] = {}
        for row in '12345678':
            for col in 'abcdefgh':
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

        for row_ind, row in zip('12345678', piece_placement):
            for col_ind, piece_str in zip('abcdefgh', row):
                if piece_str == EMPTY:
                    continue
                position = f'{col_ind}{row_ind}'
                game.place(position=position, piece_str=piece_str)

        game.complete_initialization()

        return game

    def display(self) -> None:
        print()
        for row in reversed('12345678'):
            for col in 'abcdefgh':
                position = f'{col}{row}'
                print(self.get_square(position).piece_str or ' ', end='')
            print()
        print()
        print(f'turn={self.turn}')
        print(f'castling_rights={self.castling_rights}')
        print(f'en_passant={self.en_passant}')
        print(f'half_moves={self.half_moves}')
        print(f'move_number={self.move_number}')

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


if __name__ == "__main__":
    main()
