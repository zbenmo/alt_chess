from dataclasses import dataclass
from typing import Callable, Dict, Generator, Set, Tuple


W_P, W_R, W_N, W_B, W_Q, W_K = "PRNBQK"
B_P, B_R, B_N, B_B, B_Q, B_K = "prnbqk"
EMPTY = " "

ROWS = '12345678'
COLS = 'abcdefgh'


Position = str # ex. 'e2'
PieceStr = str # ex. 'P'
Move = str # ex. 'e2e4' (UCI)


default_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
# default_fen = 'rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1'
# default_fen = 'rnbqkbnr/ppppppPp/8/8/8/8/PPPPP1PP/RNBQKBNR w KQkq - 0 1'
# default_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQkq - 0 1'


@dataclass
class Game:
    board: Dict[Position, PieceStr]
    turn: str
    castling_rights: Set[str]
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

        castling_rights = set(castling_rights)
        castling_rights.discard('-')

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
        for_black: bool = player == 'b'
        for position, piece_str in self.board.items():
            if piece_str == EMPTY:
                continue
            if piece_str.isupper() != for_black: # xor
                yield piece_str, position
