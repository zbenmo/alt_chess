from abc import ABC
from dataclasses import dataclass, field
from typing import Callable, Dict, Generator, Protocol
# from typing import Dict, Self
from events import Event


W_P, W_R, W_N, W_B, W_Q, W_K = "PRNBQK"
B_P, B_R, B_N, B_B, B_Q, B_K = "prnbqk"
EMPTY = " "


@dataclass
class Square():
    position: str
    piece_enters: Event = field(init=False, default_factory=Event)
    piece_leaves: Event = field(init=False, default_factory=Event)
    piece: str = EMPTY
    _row: int = field(init=False)
    _col: int = field(init=False)

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

    def place(self, piece: str):
        # TODO: handle existing piece
        self.piece = piece
        self.piece_enters(square=self, piece=piece)

    def pick(self):
        self.piece = EMPTY
        self.piece_leaves()

    def __repr__(self):
        return self.position


class Piece(ABC):
    def __init__(self, piece: str):
        self.piece = piece
        self.square = None
    
    def place(self, position: str, get_square: Callable[[str], Square]):
        self.square = get_square(position)


class Pawn(Piece):
    def __init__(self, piece: str):
        super().__init__(piece=piece)
        assert piece == W_P or piece == B_P
    
    def place(self, position: str, get_square: Callable[[str], Square]):
        super().place(position=position, get_square=get_square)
        direction = +1 if self.piece == W_P else -1
        starting_row = '2' if self.piece == W_P else '7'
        col, row = self.square.col, self.square.row
        self.moves = [self.square.add(rows=1 * direction, cols=0)]
        if row == starting_row:
            self.moves.append(self.square.add(rows=2 * direction, cols=0))
        self.captures = []
        if col > 'a':
            self.captures.append(self.square.add(rows=1 * direction, cols=-1))
        if col < 'h':
            self.captures.append(self.square.add(rows=1 * direction, cols=+1))
        self.square.piece_leaves += (self._pawn_leaves_square, )

    def _pawn_enters_square(self, square: Square):
        print(f'{self.piece} enters {square}')
        self.square = square
        self.square.piece_leaves += (self._pawn_leaves_square, )

    def _pawn_leaves_square(self):
        print(f'{self.piece} leaves {self.square}')
        self.square.piece_leaves.remove(self._pawn_leaves_square, )


class Knight(Piece):
    def __init__(self, piece: str):
        super().__init__(piece=piece)
        assert piece == W_N or piece == B_N

class Rook(Piece):
    def __init__(self, piece: str):
        super().__init__(piece=piece)
        assert piece == W_R or piece == B_R


class Bishop(Piece):
    def __init__(self, piece: str):
        super().__init__(piece=piece)
        assert piece == W_B or piece == B_B


class Queen(Piece):
    def __init__(self, piece: str):
        super().__init__(piece=piece)
        assert piece == W_Q or piece == B_Q


class King(Piece):
    def __init__(self, piece: str):
        super().__init__(piece=piece)
        assert piece == W_K or piece == B_K


default_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'


class Game():
    def __init__(self):
        self.board: Dict[str, Square] = {}
        for row in '12345678':
            for col in 'abcdefgh':
                position = f'{col}{row}'
                self.board[position] = Square(position=position)

    @staticmethod
    def from_fen(fen: str = default_fen) -> 'Game':
        game = Game()

        (
            position,
            turn,
            castling_rights,
            en_passant,
            half_moves,
            move_number
        ) = fen.split(' ')

        def split_and_expand(line: str) -> Generator[str, None, None]:
            for char in line:
                if char.isdigit():
                    for _ in range(int(char)):
                        yield EMPTY
                else:
                    yield char

        piece_placement = [
            list(split_and_expand(line))
            for line in reversed(position.split('/'))
        ]

        for row_ind, row in zip('12345678', piece_placement):
            for col_ind, p in zip('abcdefgh', row):
                if p == EMPTY:
                    continue
                position = f'{col_ind}{row_ind}'
                game.place(position=position, piece=Game.piece_for(p))

        return game

    @staticmethod
    def piece_for(p: str) -> Piece:
        match p:
            case 'p' | 'P':
                return Pawn(p)
            case 'r' | 'R':
                return Rook(p)
            case 'n' | 'N':
                return Knight(p)
            case 'b' | 'B':
                return Bishop(p)
            case 'q' | 'Q':
                return Queen(p)
            case 'k' | 'K':
                return King(p)
            case _:
                assert False, f'{p=}'

    def place(self, position: str, piece: Piece):
        piece.place(position=position, get_square=self.get_square)

    def get_square(self, position: str) -> Square:
        return self.board[position]




def main():
    game = Game.from_fen()

    # pawn = Pawn('P')
    # game.place('e2', pawn)
    game.get_square('e2').pick()


if __name__ == "__main__":
    main()
