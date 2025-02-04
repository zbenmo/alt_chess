from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Dict, Generator, List, Protocol, Tuple
from events import Event


W_P, W_R, W_N, W_B, W_Q, W_K = "PRNBQK"
B_P, B_R, B_N, B_B, B_Q, B_K = "prnbqk"
EMPTY = " "

ROWS = '12345678'
COLS = 'abcdefgh'

# W_EN_PASSANT = 'E'
# B_EN_PASSANT = 'e'


DEBUG_PIECES = [] # B_K]


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

    def pick(self) -> str | None:
        previous_piece, self.piece_str = self.piece_str, None 
        self.changed(square=self)
        return previous_piece

    def indicate_can_move_to(self, piece_str: str, can_capture: bool = True, can_promote: bool = False, sign=+1):
        if piece_str.isupper():
            self.white_can_move += sign
            self.white_can_capture += int(can_capture) * sign
            self.white_can_promote += int(can_promote) * sign
        else:
            self.black_can_move += sign
            self.black_can_capture += int(can_capture) * sign
            self.black_can_promote += int(can_promote) * sign

    def __repr__(self):
        return self.position


class Board(Protocol):
    def get_square(position: str) -> Square:
        ...

    @property
    def en_passant() -> str | None:
        ...
    
    @en_passant.setter
    def en_passant(en_passant_square_str: str | None) -> None:
        ...

    @property
    def white_king_position(self) -> str:
        ...
    
    @white_king_position.setter
    def white_king_position(self, position: str) -> None:
        ...

    @property
    def black_king_position(self) -> str:
        ...

    @black_king_position.setter
    def black_king_position(self, position: str) -> None:
        ...

    def count_checks(self, which_king: str) -> int:
        ...

    def take_move(self, source_sqaure_str: str, target_square_str: str, promotion: str=None) -> None:
        ...

    def undo(self) -> None:
        ...

    def place(self, position: str, piece_str: str):
        ...

    def pick(self, position: str) -> str | None:
        ...


@dataclass
class Move:
    pieces_to_remove: List[str] # positions
    pieces_to_add: Dict[str, str] # position -> piece_str
    _undo_pieces_to_remove: List[str] = field(init=False, default=None) # positions
    _undo_pieces_to_add: List[str] = field(init=False, default=None) # position -> piece_str

    def take(self, board: Board):
        self._undo_pieces_to_remove = []
        self._undo_pieces_to_add = []
        for position in self.pieces_to_remove:
            picked = board.pick(position)
            self._undo_pieces_to_add[position] = picked
        for position, piece_str in self.pieces_to_add.items():
            board.place(position, piece_str)
            self._undo_pieces_to_remove.append[position]

    def undo(self, board: Board):
        for position in self._undo_pieces_to_remove:
            _ = board.pick(position)
        for position, piece_str in self._undo_pieces_to_add.items():
            board.place(position, piece_str)
        self._undo_pieces_to_remove = None
        self._undo_pieces_to_add = None


@dataclass
class PieceEvaluation:
    control_squares: List[str] = field(default_factory=list) # potentially capture if relevant
    potential_moves: List[str] = field(default_factory=list)
    want_to_watch_squares: List[str] = field(default_factory=list)


class Piece(ABC):
    def __init__(self, piece_str: str, position: str, board: Board):
        self._piece_str = piece_str
        self._position = position
        self._board = board
        self._moves: List[str] = []
        self._piece_evaluation = PieceEvaluation()

    def evaluate(self):
        piece_evaluation = self.evaluate_disregarding_king_threats()

        previous_potential_moves = set(self._piece_evaluation.potential_moves)
        current_potential_moves = set(piece_evaluation.potential_moves)

        previous_control_squares = set(self._piece_evaluation.control_squares)
        current_control_squares = set(piece_evaluation.control_squares)

        for potential_move in (
            (current_potential_moves | current_control_squares)
            -
            (previous_potential_moves | previous_control_squares)
        ):
            target_square = self._board.get_square(potential_move) 
            target_square.indicate_can_move_to(self._piece_str, can_capture=potential_move in current_control_squares)

        for potential_move in (
            (previous_potential_moves | previous_control_squares)
            -
            (current_potential_moves | current_control_squares)
            ):
            target_square = self._board.get_square(potential_move) 
            target_square.indicate_can_move_to(self._piece_str, can_capture=potential_move in previous_control_squares, sign=-1)

        self._piece_evaluation.potential_moves = list(current_potential_moves)
        self._piece_evaluation.control_squares = list(current_control_squares)

        previous_want_to_watch_squares = set(self._piece_evaluation.want_to_watch_squares)
        current_want_to_watch_squares = set(piece_evaluation.want_to_watch_squares)

        for watch_sqaure in current_want_to_watch_squares - previous_want_to_watch_squares:
            self._register_for_a_square_change(watch_sqaure)
        for watch_square in previous_want_to_watch_squares - current_want_to_watch_squares:
            self._unregister_from_a_square_change(watch_square)

        self._piece_evaluation.want_to_watch_squares = list(current_want_to_watch_squares)

    @abstractmethod
    def evaluate_disregarding_king_threats(self) -> PieceEvaluation:
        pass

    def prepare_available_moves(self) -> List[str]:
        moves = []
        my_king = W_K if self.same_color(W_K) else B_K
        self.evaluate()
        if self._piece_str in DEBUG_PIECES:
            print(f'prepare_available_moves: {self}, _piece_evaluation={self._piece_evaluation}')
        for target_square_str in self._piece_evaluation.potential_moves:
            self._board.take_move(self._position, target_square_str) # TODO: more interesting moves....
            if self._board.count_checks(my_king) < 1:
                moves.append(f'{self._position}{target_square_str}')
            self._board.undo()
        return moves

    def same_color(self, other_piece_str: str) -> bool:
        return self._piece_str.isupper() == other_piece_str.isupper()

    def _register_for_a_square_change(self, position: str):
        square = self._board.get_square(position)
        square.changed += (self._square_changed, )
        # self._piece_evaluation.want_to_watch_squares is adjusted above

    def _unregister_from_a_square_change(self, position: str):
        square = self._board.get_square(position)
        square.changed.remove(self._square_changed)
        # self._piece_evaluation.want_to_watch_squares is adjusted above

    def _square_changed(self, square: Square):
        self.evaluate()

    def clean(self) -> None:
        for position in self._piece_evaluation.want_to_watch_squares:
            square = self._board.get_square(position)
            square.changed.remove(self._square_changed)
        self._piece_evaluation.want_to_watch_squares.clear()

    def __repr__(self):
        return f'{self._piece_str} at {self._position}'


class Pawn(Piece):
    def __init__(self, piece_str: str, position: str, board: Board):
        super().__init__(piece_str=piece_str, position=position, board=board)
        assert piece_str == W_P or piece_str == B_P
        square = self._board.get_square(self._position)
        direction = +1 if self._piece_str == W_P else -1
        starting_row = '2' if self._piece_str == W_P else '7'
        pre_promotion_row = '7' if self._piece_str == W_P else '2'
        col, row = square.col, square.row
        self.pre_promotion = pre_promotion_row == row
        square = self._board.get_square(self._position)
        self._theoretical_moves = [square.add(rows=1 * direction, cols=0)]
        if row == starting_row:
            self._theoretical_moves.append(square.add(rows=2 * direction, cols=0))
        self._theoretical_captures = []
        if col > COLS[0]:
            self._theoretical_captures.append(square.add(rows=1 * direction, cols=-1))
        if col < COLS[-1]:
            self._theoretical_captures.append(square.add(rows=1 * direction, cols=+1))
    
    def evaluate_disregarding_king_threats(self):
        piece_evaluation = PieceEvaluation()
        for move_square_str in self._theoretical_moves:
            piece_evaluation.want_to_watch_squares.append(move_square_str)
            move_square = self._board.get_square(move_square_str)
            if move_square.piece_str is None:
                piece_evaluation.potential_moves.append(move_square_str)
            else:
                break
        for capture_square_str in self._theoretical_captures:
            piece_evaluation.want_to_watch_squares.append(capture_square_str)
            piece_evaluation.control_squares.append(capture_square_str)
            capture_square = self._board.get_square(capture_square_str)
            if capture_square.piece_str is None:
                if self._board.en_passant == capture_square_str:
                    piece_evaluation.potential_moves.append(capture_square_str) # en-passant
                # otherwise this move is not relevant at the moment
            elif not self.same_color(capture_square.piece_str):
                piece_evaluation.potential_moves.append(capture_square_str) # capture
        return piece_evaluation

    def prepare_available_moves(self) -> List[str]:
        if not self.pre_promotion:
            return super().prepare_available_moves()
        moves = []
        my_king = W_K if self.same_color(W_K) else B_K
        self.evaluate()
        promotions = [W_Q, W_R, W_B, W_N] if self._piece_str == W_P else [B_Q, B_R, B_B, B_N]
        if self._piece_str in DEBUG_PIECES:
            print(f'prepare_available_moves: {self}, _piece_evaluation={self._piece_evaluation}')
        for target_square_str, promotion in product(self._piece_evaluation.potential_moves, promotions):
            self._board.take_move(self._position, target_square_str, promotion=promotion)
            if self._board.count_checks(my_king) < 1:
                moves.append(f'{self._position}{target_square_str}{promotion}')
            self._board.undo()
        return moves


class Knight(Piece):
    def __init__(self, piece_str: str, position: str, board: Board):
        super().__init__(piece_str=piece_str, position=position, board=board)
        assert piece_str == W_N or piece_str == B_N
        square = self._board.get_square(self._position)
        self._theoretical_moves = [
            position
            for add_row, add_col in zip([1, 2], [2, 1])
            for sign_row, sign_col in product([-1, 1], [-1, 1])
            if (position := square.add(rows=sign_row * add_row, cols=sign_col * add_col)) is not None
        ]

    def evaluate_disregarding_king_threats(self) -> PieceEvaluation:
        piece_evaluation = PieceEvaluation()
        for target_square_str in self._theoretical_moves:
            piece_evaluation.want_to_watch_squares.append(target_square_str)
            piece_evaluation.control_squares.append(target_square_str)
            target_square = self._board.get_square(target_square_str)
            if target_square.piece_str is None:
                piece_evaluation.potential_moves.append(target_square_str)
            elif not self.same_color(target_square.piece_str):
                piece_evaluation.potential_moves.append(target_square_str)
        return piece_evaluation


class RayBased(Piece, ABC):
    def __init__(self, piece_str: str, position: str, board: Board):
        super().__init__(piece_str=piece_str, position=position, board=board)
        self._rays = [
            [target_square_str for target_square_str in ray if target_square_str]
            # if a target_square_str is "outside of the board" we shall have it here as None
            for ray in self.get_rays()
        ]

    @abstractmethod
    def get_rays(self) -> List[List[str]]:
        pass

    def evaluate_disregarding_king_threats(self) -> PieceEvaluation:
        piece_evaluation = PieceEvaluation()
        for ray in self._rays:
            for target_square_str in ray:
                piece_evaluation.want_to_watch_squares.append(target_square_str)
                piece_evaluation.control_squares.append(target_square_str)
                target_square = self._board.get_square(target_square_str)
                if target_square.piece_str is None:
                    piece_evaluation.potential_moves.append(target_square_str)
                else:
                    if not self.same_color(target_square.piece_str):
                        piece_evaluation.potential_moves.append(target_square_str) # capture
                    break # this ray reached a piece
        return piece_evaluation


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
        castling_rights = self._board.castling_rights
        queen_side = ('Q' if piece_str == W_R else 'q') in castling_rights
        king_side = ('K' if piece_str == W_R else 'k') in castling_rights
        row, col = self._position.row, self._position.col
        self._relevant_for_king_side_castling = (
            king_side and (col == COLS[-1]) and (row == (ROWS[0] if piece_str == W_R else ROWS[-1]))
        )
        self._relevant_for_queen_side_castling = (
            queen_side and (col == COLS[0]) and (row == (ROWS[0] if piece_str == W_R else ROWS[-1]))
        )

    def get_rays(self) -> List[List[str]]:
        return self._get_rook_rays()

    def clean(self) -> None:
        super().clear()
        # TODO: how to take back??????
        if self._relevant_for_king_side_castling:
            self._board.castling_rights.remove(('K' if self._piece_str == W_R else 'k'))
        if self._relevant_for_queen_side_castling:
            self._board.castling_rights.remove(('Q' if self._piece_str == W_R else 'q'))


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
        # update the king's position
        if piece_str == W_K:
            self._board.white_king_position = self._position
        else:
            self._board.black_king_position = self._position
        # preparation
        square = self._board.get_square(self._position)
        self._theoretical_moves = [
            position
            for add_rows, add_cols in product([-1, 0, 1], [-1, 0, 1]) if (
                (add_rows != 0 or add_cols != 0)
                and
                ((position := square.add(add_rows, add_cols)) is not None)
            )
        ]
        if self._piece_str in DEBUG_PIECES:
            print(f'{self}, _theoretical_moves={self._theoretical_moves}')

    def evaluate_disregarding_king_threats(self) -> PieceEvaluation:
        piece_evaluation = PieceEvaluation()
        for target_square_str in self._theoretical_moves:
            piece_evaluation.want_to_watch_squares.append(target_square_str)
            piece_evaluation.control_squares.append(target_square_str)
            target_square = self._board.get_square(target_square_str)
            if target_square.piece_str is None:
                piece_evaluation.potential_moves.append(target_square_str) # move to empty
            elif not self.same_color(target_square.piece_str):
                piece_evaluation.potential_moves.append(target_square_str) # capture
            else:
                pass
        if self._piece_str in DEBUG_PIECES:
            print(f'evaluate_disregarding_king_threats: {self}, piece_evaluation={piece_evaluation}')
        return piece_evaluation


# default_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
# default_fen = 'rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1'
default_fen = 'rnbqkbnr/ppppppPp/8/8/8/8/PPPPP1PP/RNBQKBNR w KQkq - 0 1'


class Game():
    def __init__(self,
                 piece_placement: List[List[str]],
                 turn: str,
                 castling_rights: str,
                 en_passant: str,
                 half_moves: int,
                 move_number: int
    ):
        assert turn in ["w", "b"], f'{turn=}'
        self.turn = turn
        self.castling_rights = castling_rights # for example: 'KQkq'
        self._en_passant = en_passant # for example 'e3'
        self.half_moves = half_moves
        self.move_number = move_number

        self.board: Dict[str, Square] = {}
        for row, col in product(ROWS, COLS):
            position = f'{col}{row}'
            self.board[position] = Square(position=position)

        self.pieces = {}
        self._white_king_position = None
        self._black_king_position = None
        self._undo_stack = []
        for row_ind, row in zip(ROWS, piece_placement):
            for col_ind, piece_str in zip(COLS, row):
                if piece_str == EMPTY:
                    continue
                position = f'{col_ind}{row_ind}'
                self.place(position=position, piece_str=piece_str)

        for piece in self.pieces.values():
            piece.evaluate()

    @property
    def en_passant(self) -> str:
        return self._en_passant
    
    @en_passant.setter
    def en_passant(self, en_passant_square_str: str) -> None:
        self._en_passant=en_passant_square_str

    @property
    def white_king_position(self) -> str:
        return self._white_king_position
    
    @white_king_position.setter
    def white_king_position(self, position: str) -> None:
        self._white_king_position = position

    @property
    def black_king_position(self) -> str:
        return self._black_king_position

    @black_king_position.setter
    def black_king_position(self, position: str) -> None:
        self._black_king_position = position

    def count_checks(self, which_king: str) -> int:
        "anything above 0 is a check"
        if which_king == W_K:
            king_position = self.white_king_position
            return self.get_square(king_position).black_can_capture
        else:
            assert which_king == B_K, f'{which_king=}'
            king_position = self.black_king_position
            return self.get_square(king_position).white_can_capture

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

        return Game(
            piece_placement=piece_placement,
            turn=turn,
            castling_rights=castling_rights,
            en_passant=en_passant,
            half_moves=half_moves,
            move_number=move_number
        )

    def _display_board(self, display_what: Callable[[Square], str]) -> None:
        for row in reversed(ROWS):
            for col in COLS:
                position = f'{col}{row}'
                square = self.get_square(position)
                print(display_what(square), end='')
            print()

    def _display_board_simple(self) -> None:
        self._display_board(lambda square: square.piece_str or ('*' if square.position == self.en_passant else '.'))

    def display(self) -> None:
        print()
        self._display_board(lambda square: square.piece_str or ('*' if square.position == self.en_passant else '.'))
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
        square = self.get_square(position)
        square.piece_str = piece_str
        piece = Game.piece_for(piece_str=piece_str, position=position, board=self)
        self.pieces[position] = piece
        piece.evaluate()

    def pick(self, position: str) -> str | None:
        square = self.get_square(position)
        piece_str = square.pick()
        if piece_str:
            piece = self.pieces.pop(position)
            assert piece._piece_str == piece_str
            piece.detach()
            del piece
        return piece_str

    def get_square(self, position: str) -> Square:
        return self.board[position]

    def take_move(self, source_sqaure_str: str, target_square_str: str, promotion: str=None) -> None:
        move = []

        piece_str = self.pick(source_sqaure_str)
        assert piece_str, f'{source_sqaure_str=}'

        target_piece_str = self.pick(target_square_str)

        to_place_piece_str = promotion if promotion else piece_str

        self.place(target_square_str, to_place_piece_str)

        move = [(piece_str, source_sqaure_str), (target_piece_str, target_square_str)]
        self._undo_stack.append(move)

    def undo(self) -> None:
        if len(self._undo_stack) < 1:
            return
        move = self._undo_stack.pop() # last
        for piece_str, position in reversed(move):

            # piece_old = self.pieces.pop(position, None)
            # if piece_old:
            #     piece_old.detach()
            #     del piece_old

            if piece_str:
                self.place(piece_str=piece_str, position=position) # -> event
            else:
                picked = self.pick(position)


def main():
    game = Game.from_fen()

    game.display()

    print(f'count_checks - white={game.count_checks(W_K)}')
    print(f'count_checks - black={game.count_checks(B_K)}')
    print()

    pieces = list(game.pieces.values())

    for piece in pieces:
        moves = piece.prepare_available_moves()
        print(f'{piece}, moves={moves}')
    print()


if __name__ == "__main__":
    main()
