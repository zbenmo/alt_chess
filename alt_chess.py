from abc import ABC
from typing import Generator, List, Tuple


W_P, W_R, W_N, W_B, W_Q, W_K = "PRNBQK"
B_P, B_R, B_N, B_B, B_Q, B_K = "prnbqk"
EMPTY = " "


class SquarePosition: # forward declaration
    pass


SquareRef = Tuple[int, int]


class SquarePosition:
    """Holds values such as 'e2'.
    Yet it is also helpful when we want to refer to the values as indices in a matix.
    """
    def __init__(self, position: str):
        self._position = position
        col, row = position
        self._col = ord(col) - ord('a') # 0 based
        self._row = int(row) - 1 # 0 based

    @property
    def row(self) -> str:
        return self._position[1]

    @property
    def col(self) -> str:
        return self._position[0]

    def add(self, rows: int, cols: int) -> SquarePosition:
        row = self._row + rows
        col = self._col + cols
        return SquarePosition(f'{chr(col + ord("a"))}{row + 1}')

    def as_square_ref(self) -> SquareRef:
        return self._row, self._col
    
    def __eq__(self, other: SquarePosition) -> bool:
        return self._position == other._position

    def __eq__(self, position: str) -> bool:
        return self._position == position

    def __hash__(self) -> int:
        return hash(self._position)

    def __repr__(self):
        return self._position


class Move(ABC):
    def __init__(self):
        pass

    def applicable(self, board) -> bool:
        pass

    def apply(self, board):
        pass


class SimpleMove(Move):
    def __init__(self, from_position: SquarePosition, to_position: SquarePosition, piece: str, if_free=True, if_capture=True):
        super().__init__()
        self.from_position = from_position
        self.to_position = to_position
        self.piece = piece
        self.if_free = if_free
        self.if_capture = if_capture

    def applicable(self, board) -> bool:
        return ((
                    self.if_free and board[self.to_position].free()
                )
                or
                (
                    self.if_capture and board.capturable(self.piece)
                )
        )

    def apply(self, board):
        board[self.to_position].pick()
        board[self.from_position].pick()
        board[self.to_position].place(self.piece)


class PromotionMove(Move): # potential to inherit from SimpleMove TODO: ?
    def __init__(self, from_position: SquarePosition, to_position: SquarePosition, piece: str, if_free: bool = True, if_capture: bool = True, promotion: str = None):
        super().__init__()
        self.from_position = from_position
        self.to_position = to_position
        self.piece = piece
        self.if_free = if_free
        self.if_capture = if_capture
        self.promotion = promotion

    def applicable(self, board) -> bool:
        return ((
                    self.if_free and board[self.to_position].free()
                )
                or
                (
                    self.if_capture and board.capturable(self.piece)
                )
        )

    def apply(self, board):
        board[self.to_position].pick()
        board[self.from_position].pick()
        board[self.to_position].place(self.promotion)

# class PawnAdvanceMove(SimpleMove):
#     def __init__(self, from_position, to_position, piece):
#         super().__init__(from_position, to_position, piece)

#     def applicable(self, board) -> bool:
#         return board[self.to_position].free()


# class PawnCaptureMove(SimpleMove):
#     def __init__(self, from_position, to_position, piece):
#         super().__init__(from_position, to_position, piece)

#     def applicable(self, board) -> bool:
#         return board[self.to_position].capturable(self.piece)


# class PawnDoubleAdvance(Move):
#     def __init__(self, from_position, to_position, piece):
#         super().__init__(from_position, to_position, piece)


# class PawnPromotion(Move):
#     def __init__(self, from_position, to_position, piece):
#         super().__init__(from_position, to_position, piece)


class Square:
    def __init__(self, position: SquarePosition):
        self.position = position
        self.moves = []

    def add_move(self, move):
        self.moves.append(move)

    def get_moves(self):
        return self.moves

    def place(self, piece: str):
        if piece == 'P':
            self.add_move(
                    SimpleMove(
                        from_position=self.position,
                        to_position=self.position.add(1, 0),
                        piece=piece,
                        if_free=True,
                        if_capture=False
                    )
                )
            if self.position.col != 'a':
                self.add_move(
                        SimpleMove(
                            from_position=self.position,
                            to_position=self.position.add(1, -1),
                            piece=piece,
                            if_free=False,
                            if_capture=True
                        )
                    )
            if self.position.col != 'h':
                self.add_move(
                        SimpleMove(
                            from_position=self.position,
                            to_position=self.position.add(1, 1),
                            piece=piece,
                            if_free=False,
                            if_capture=True
                        )
                    )

        # if self.row == '2' and piece == 'P':
        #     print("potential for double advance")
        #         board.add_move(row_col_ind_to_square(row_i, col_i), row_col_ind_to_square(row_i + 1, col_i), 'P')
        #         if row_i == 1:
        #             board.add_move(row_col_ind_to_square(row_i, col_i), row_col_ind_to_square(3, col_i), 'P')


class AllowDoubleAdvanceSquare(Square):
    def __init__(self, position: SquarePosition, for_piece: str):
        super().__init__(position)
        self.for_piece = for_piece

    def place(self, piece: str):
        super().place(piece)
        if piece != self.for_piece:
            return

        direction = -1 if piece.islower() else +1

        self.add_move(
            SimpleMove(
                from_position=self.position,
                to_position=self.position.add(2 * direction, 0),
                piece=piece,
                if_free=True,
                if_capture=False,
            )
            # TODO: how to guarantee also free on the way
            # TODO: how to mark en-passant ?
        )

class PrePromotionSquare(Square):
    def __init__(self, position: SquarePosition, for_piece: str):
        super().__init__(position)
        self.for_piece = for_piece

    def place(self, piece: str):
        if piece != self.for_piece:
            super().place(piece)
            return

        if piece.islower():
            options = 'qrbn'
            direction = -1
        else:
            options = 'QRBN'
            direction = +1

        for promotion in options:
            self.add_move(
                    PromotionMove(
                        from_position=self.position,
                        to_position=self.position.add(direction, 0),
                        piece=piece,
                        if_free=True,
                        if_capture=False,
                        promotion=promotion
                    )
                )
            if self.position.col != 'a':
                self.add_move(
                        PromotionMove(
                            from_position=self.position,
                            to_position=self.position.add(direction, -1),
                            piece=piece,
                            if_free=False,
                            if_capture=True,
                            promotion=promotion
                        )
                    )
            if self.position.col != 'h':
                self.add_move(
                        PromotionMove(
                            from_position=self.position,
                            to_position=self.position.add(direction, 1),
                            piece=piece,
                            if_free=False,
                            if_capture=True,
                            promotion=promotion
                        )
                    )


class Move:
    def __init__(self, piece, start, end):
        self.piece = piece
        self.start = start
        self.end = end


class Board:
    def __init__(self):
        self.squares = {}
        for row in '12345678':
            for col in 'abcdefgh':
                position = SquarePosition(f'{col}{row}')
                if row == '7':
                    square = PrePromotionSquare(position=position, for_piece='P')
                elif row == '2':
                    square = PrePromotionSquare(position=position, for_piece='p')
                else:
                    square = Square(position=position)
                self.squares[position] = square

    def add_move(self, start, end, piece):
        move = SimpleMove(start, end, piece)
        self.squares[start].add_move(move)

    def get_moves(self, position):
        return self.squares[position].get_moves()

    def get_all_moves(self) -> Generator[Move, None, None]:
        for square in self.squares.values():
            yield from square.get_moves()

    def apply_move(self, move: Move):
        pass


default_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'


def board_from_fen(fen: str):
    (
        position,
        turn,
        castling_rights,
        en_passant,
        half_moves,
        move_number
    ) = fen.split(' ')

    board = Board()

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

    square_position = SquarePosition('a1')
    for row in piece_placement:
        for piece in row:
            if piece != EMPTY:
                board.squares[square_position].place(piece)
            square_position = square_position.add(0, 1)
        square_position = square_position.add(1, -8)

    return board


def main():
    board = board_from_fen(default_fen)

    e2 = SquarePosition('e2')

    for move in board.get_moves(e2):
        print(f"{move.piece} from {move.from_position} to {move.to_position}")

    print('-' * 80)

    for move in board.get_all_moves():
        print(f"{move.piece} from {move.from_position} to {move.to_position}")


if __name__ == "__main__":
    main()
