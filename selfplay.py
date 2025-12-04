import chess
import torch
import numpy as np
from agent import RadicalAgent
from config import Config

def chess_board_to_tensor(board):
    """
    Converts a chess.Board object into a 12x8x8 tensor.
    Channels: 6 pieces (P, N, B, R, Q, K) * 2 colors (White, Black)
    """
    # Map pieces to channels 0-11
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, 
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    # Initialize 12x8x8 tensor
    tensor = torch.zeros(12, 8, 8)
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Determine channel offset: White=0, Black=6
            offset = 0 if piece.color == chess.WHITE else 6
            channel = piece_map[piece.piece_type] + offset
            
            # Convert square index (0-63) to row/col
            row = 7 - (square // 8) # Rank 8 is row 0
            col = square % 8
            
            tensor[channel, row, col] = 1.0
            
    return tensor.unsqueeze(0) # Add batch dimension (1, 12, 8, 8)

def move_to_index(move):
    """
    Converts a chess.Move object to an index in the 4096 action space.
    Action = From_Square * 64 + To_Square
    """
    return move.from_square * 64 + move.to_square

def index_to_move(index):
    """
    Converts an action index back to a chess.Move object.
    """
    from_sq = index // 64
    to_sq = index % 64
    return chess.Move(from_sq, to_sq)

def run_self_play_game(agent_white, agent_black, max_moves=100):
    board = chess.Board()
    print("Starting new game...")
    
    for move_count in range(max_moves):
        if board.is_game_over():
            break
            
        # 1. Observe
        state_tensor = chess_board_to_tensor(board)
        if board.turn == chess.WHITE:
            agent_white.observe(state_tensor)
        else:
            agent_black.observe(state_tensor)
        
        # 2. Think & Act
        # Create a mask of -inf
        legal_moves = list(board.legal_moves)
        legal_indices = [move_to_index(m) for m in legal_moves]
        
        # We need a mask of shape (1, 4096)
        mask = torch.full((1, Config.ACTION_SPACE), -float('inf'))
        mask[0, legal_indices] = 0 # Unmask legal moves
        
        if board.turn == chess.WHITE:
            action_index, status = agent_white.act(action_mask=mask)
        else:
            action_index, status = agent_black.act(action_mask=mask)
        
        move = index_to_move(action_index)
        board.push(move)
        
        # print(f"Move {move_count+1}: {move}")
        
    print(f"Game Over. Result: {board.result()}")
    print(board)

if __name__ == "__main__":
    agent_white = RadicalAgent()
    agent_black = RadicalAgent()
    print("Agent initialized.")
    run_self_play_game(agent_white, agent_black)
