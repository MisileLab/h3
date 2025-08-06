"""Monte Carlo Tree Search implementation."""

import math
import time
from typing import Dict, List, Tuple, Optional, Any

import chess
import numpy as np
import torch

from adela.core.board import BoardRepresentation
from adela.gating.system import MixtureOfExperts


class Node:
    """Node in the MCTS tree."""

    def __init__(
        self, 
        board: BoardRepresentation,
        parent: Optional["Node"] = None,
        move: Optional[chess.Move] = None,
        prior: float = 0.0
    ) -> None:
        """Initialize a node.

        Args:
            board: Board representation.
            parent: Parent node.
            move: Move that led to this node.
            prior: Prior probability of this node.
        """
        self.board = board
        self.parent = parent
        self.move = move
        self.prior = prior
        
        self.children: Dict[chess.Move, "Node"] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.expanded = False
        
    def value(self) -> float:
        """Get the value of this node.

        Returns:
            Node value.
        """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def expand(
        self, 
        policy_logits: np.ndarray
    ) -> None:
        """Expand the node with policy logits.

        Args:
            policy_logits: Policy logits for legal moves.
        """
        legal_moves = self.board.get_legal_moves()
        
        # Create children for each legal move
        for i, move in enumerate(legal_moves):
            # Create a new board for the child
            child_board = BoardRepresentation(self.board.get_fen())
            child_board.make_move(move)
            
            # Get the prior probability for this move
            prior = policy_logits[i]
            
            # Create child node
            self.children[move] = Node(
                board=child_board,
                parent=self,
                move=move,
                prior=prior
            )
        
        self.expanded = True
    
    def select_child(self, c_puct: float = 1.0) -> "Node":
        """Select a child node according to the PUCT algorithm.

        Args:
            c_puct: Exploration constant.

        Returns:
            Selected child node.
        """
        # Find the child with the highest UCB score
        best_score = float("-inf")
        best_child = None
        
        # Total visit count for parent
        parent_visit_count = self.visit_count
        
        for child in self.children.values():
            # UCB score
            if child.visit_count > 0:
                # Exploitation term
                q_value = child.value()
                # Exploration term
                u_value = c_puct * child.prior * math.sqrt(parent_visit_count) / (1 + child.visit_count)
                # PUCT score
                score = q_value + u_value
            else:
                # If the child has not been visited, prioritize it
                score = c_puct * child.prior * math.sqrt(parent_visit_count + 1e-8)
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def update(self, value: float) -> None:
        """Update the node with a value.

        Args:
            value: Value to update with.
        """
        self.visit_count += 1
        self.value_sum += value
    
    def is_leaf(self) -> bool:
        """Check if the node is a leaf node.

        Returns:
            True if the node is a leaf node, False otherwise.
        """
        return len(self.children) == 0
    
    def is_terminal(self) -> bool:
        """Check if the node is a terminal node.

        Returns:
            True if the node is a terminal node, False otherwise.
        """
        return self.board.is_game_over()


class MCTS:
    """Monte Carlo Tree Search algorithm."""

    def __init__(
        self, 
        model: MixtureOfExperts,
        num_simulations: int = 800,
        c_puct: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_weight: float = 0.25,
        temperature: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        """Initialize the MCTS algorithm.

        Args:
            model: Neural network model.
            num_simulations: Number of simulations to run.
            c_puct: Exploration constant.
            dirichlet_alpha: Dirichlet noise alpha parameter.
            dirichlet_weight: Weight of Dirichlet noise.
            temperature: Temperature for move selection.
            device: Device to run the model on.
        """
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        self.temperature = temperature
        self.device = device
        
    def search(
        self, 
        board: BoardRepresentation,
        time_limit: Optional[float] = None
    ) -> Dict[chess.Move, float]:
        """Run the MCTS algorithm.

        Args:
            board: Initial board state.
            time_limit: Optional time limit in seconds.

        Returns:
            Dictionary mapping moves to visit counts.
        """
        # Create root node
        root = Node(board)
        
        # Get policy and value from the model
        board_tensor, additional_features = self._prepare_input(board)
        policy_logits, value, _ = self.model(board_tensor, additional_features)
        
        # Convert to numpy
        policy_logits = policy_logits[0].detach().cpu().numpy()
        
        # Add Dirichlet noise to the root
        legal_moves = board.get_legal_moves()
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))
        
        # Apply noise to policy
        for i, move in enumerate(legal_moves):
            policy_logits[i] = (1 - self.dirichlet_weight) * policy_logits[i] + self.dirichlet_weight * noise[i]
        
        # Normalize policy
        policy_sum = np.sum(policy_logits)
        if policy_sum > 0:
            policy_logits /= policy_sum
        
        # Expand root node
        root.expand(policy_logits)
        
        # Start search
        start_time = time.time()
        for i in range(self.num_simulations):
            # Check time limit
            if time_limit and time.time() - start_time > time_limit:
                break
            
            # Selection
            node = root
            search_path = [node]
            
            while node.expanded and not node.is_terminal():
                node = node.select_child(self.c_puct)
                search_path.append(node)
            
            # Expansion and evaluation
            if not node.is_terminal():
                # Get policy and value from the model
                board_tensor, additional_features = self._prepare_input(node.board)
                policy_logits, value, _ = self.model(board_tensor, additional_features)
                
                # Convert to numpy
                policy_logits = policy_logits[0].detach().cpu().numpy()
                value = value[0, 0].item()
                
                # Expand the node
                node.expand(policy_logits)
            else:
                # Terminal node, use the game result as value
                result = node.board.get_result()
                if result == "1-0":
                    value = 1.0
                elif result == "0-1":
                    value = -1.0
                else:  # Draw
                    value = 0.0
                
                # Adjust value based on the side to move
                if not node.board.board.turn:  # If it's black's turn
                    value = -value
            
            # Backpropagation
            for node in reversed(search_path):
                # Update value (from the perspective of the current player)
                node.update(value)
                # Flip the value for the next node (parent's perspective)
                value = -value
        
        # Return move probabilities based on visit counts
        visit_counts = {
            move: child.visit_count
            for move, child in root.children.items()
        }
        
        return visit_counts
    
    def _prepare_input(
        self, 
        board: BoardRepresentation
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare input tensors for the model.

        Args:
            board: Board representation.

        Returns:
            Tuple of (board_tensor, additional_features).
        """
        # Get board tensor and additional features
        board_tensor = board.get_board_tensor()
        additional_features = board.get_additional_features()
        
        # Convert to torch tensors
        board_tensor = torch.tensor(board_tensor, dtype=torch.float32, device=self.device).unsqueeze(0)
        additional_features = torch.tensor(additional_features, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        return board_tensor, additional_features
    
    def get_best_move(
        self, 
        board: BoardRepresentation,
        temperature: Optional[float] = None
    ) -> chess.Move:
        """Get the best move according to the MCTS algorithm.

        Args:
            board: Board representation.
            temperature: Temperature for move selection. If None, uses self.temperature.

        Returns:
            Best move.
        """
        if temperature is None:
            temperature = self.temperature
        
        # Run MCTS
        visit_counts = self.search(board)
        
        # Get moves and visit counts
        moves = list(visit_counts.keys())
        counts = np.array([visit_counts[move] for move in moves])
        
        # Apply temperature
        if temperature == 0:
            # Deterministic selection
            best_idx = np.argmax(counts)
            return moves[best_idx]
        else:
            # Stochastic selection
            counts = counts ** (1 / temperature)
            counts = counts / np.sum(counts)
            selected_idx = np.random.choice(len(moves), p=counts)
            return moves[selected_idx]
