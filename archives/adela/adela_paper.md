# Adela: A Chess Engine with a Mixture of Experts and Self-Training

## Abstract

This paper introduces Adela, a novel chess engine that leverages a Mixture of Experts (MoE) architecture to enhance its playing strength and adaptability. Adela's core is an MoE model that combines the knowledge of specialized experts in different aspects of the game, including game phase (opening, middlegame, and endgame), playing style (tactical, positional, attacking, and defensive), and opponent adaptation (anti-engine, anti-human, and counter-style). A dynamic gating network selects the most suitable expert(s) for each board position, allowing the engine to adapt its strategy in real-time. Adela uses Monte Carlo Tree Search (MCTS) for move selection, guided by the policy and value outputs of the MoE model. The engine is trained using a self-training pipeline that continuously generates training data from self-play games, enabling progressive improvement. This paper details Adela's architecture, the self-training process, and discusses its potential for achieving high-level chess performance.

## 1. Introduction

The development of superhuman chess engines has been a long-standing challenge in the field of artificial intelligence. From Deep Blue's historic victory over Garry Kasparov to the recent dominance of AlphaZero and Leela Chess Zero, the quest for the perfect chess-playing machine has driven significant advancements in AI research. Modern chess engines, particularly those based on neural networks, have reached unprecedented levels of performance, often relying on a single, monolithic model to evaluate positions and select moves.

While this approach has proven to be highly effective, it has its limitations. A single model may not be optimal for all phases of the game, as the strategic considerations in the opening, middlegame, and endgame can be vastly different. Similarly, a monolithic model may struggle to adapt to different playing styles or to exploit the weaknesses of specific opponents.

To address these limitations, we present Adela, a new chess engine that employs a Mixture of Experts (MoE) architecture. The core idea behind Adela is to divide the complex task of chess playing into smaller, more manageable sub-problems, each handled by a specialized expert network. These experts are trained to excel in specific areas of the game, such as different game phases, playing styles, or opponent types. A gating network then learns to dynamically combine the outputs of these experts, selecting the most appropriate strategy for any given position.

Adela's MoE model is integrated with a Monte Carlo Tree Search (MCTS) algorithm, which explores the search space of possible moves and guides the decision-making process. The engine is trained using a self-training pipeline, where it learns by playing against itself and generating its own training data. This allows Adela to continuously improve its performance without the need for human-annotated data.

This paper makes the following key contributions:

*   **A novel Mixture of Experts (MoE) architecture for chess:** We introduce a new MoE model that combines specialized experts for game phase, playing style, and opponent adaptation.
*   **A dynamic gating mechanism for expert selection:** We describe a gating network that learns to select the most appropriate expert(s) for each position.
*   **A self-training pipeline for continuous improvement:** We detail a self-training process that enables the engine to learn from self-play and progressively enhance its playing strength.

The remainder of this paper is organized as follows: Section 2 describes Adela's system architecture in detail. Section 3 explains the self-training pipeline. Section 4 presents our experimental results and analysis. Finally, Section 5 concludes the paper and discusses future research directions.

## 2. System Architecture

Adela's architecture is designed to be modular and extensible, with the MoE model at its core. This section provides a detailed overview of the system's components, including the MoE model, the specialized experts, and the MCTS algorithm.

### 2.1. Mixture of Experts (MoE) Model

The MoE model is a powerful ensemble learning technique that combines the predictions of multiple expert models. In Adela, the MoE model consists of a set of specialized expert networks and a gating network that learns to assign weights to each expert's output based on the current board position.

The gating network is a feed-forward neural network that takes the board state and other relevant features as input. The board state is represented as a tensor of shape (12, 8, 8), where the 12 channels represent the different piece types for each color. The additional features include information about the game state, such as castling rights, en passant squares, and the number of moves since the last capture or pawn move. The gating network outputs a probability distribution over the experts, representing the confidence in each expert's ability to handle the current position.

The final output of the MoE model is a weighted combination of the outputs of all the experts. This allows the model to dynamically adapt its strategy by relying on the most relevant experts for each position.

### 2.2. Specialized Experts

Adela's MoE model includes three types of specialized experts:

*   **Phase Experts:** These experts are specialized for the different phases of the game: opening, middlegame, and endgame. Each phase expert is a convolutional neural network (CNN) that has been trained on a large dataset of games from that specific phase. This allows the experts to learn the specific patterns and strategies that are most relevant to each phase.
*   **Style Experts:** These experts are specialized for different playing styles: tactical, positional, attacking, and defensive. Each style expert is a CNN that has been trained to recognize and respond to positions that are characteristic of that particular style. This allows the engine to adapt its playing style to the demands of the position.
*   **Adaptation Experts:** These experts are specialized for adapting to different opponent types: anti-engine, anti-human, and counter-style. Each adaptation expert is a CNN that has been trained to recognize and exploit the weaknesses of a particular opponent type. This allows the engine to learn from its opponents and to adjust its strategy accordingly.

Each expert is a convolutional neural network (CNN) with a similar architecture, consisting of a series of residual blocks followed by a policy head and a value head. The policy head outputs a probability distribution over all possible moves, while the value head outputs a scalar value that represents the evaluation of the current position.

### 2.3. Monte Carlo Tree Search (MCTS)

Adela uses MCTS to search for the best move. MCTS is a heuristic search algorithm that has been successfully applied to a wide range of games, including chess. The algorithm builds a search tree by simulating games from the current position. The results of these simulations are then used to update the values of the nodes in the tree, which in turn guide the search towards the most promising moves.

The MoE model is integrated with MCTS in two ways:

*   **Policy Head:** The policy head of the MoE model is used to guide the selection of moves during the simulation phase of MCTS. This allows the search to focus on the most promising moves, which significantly improves the efficiency of the algorithm.
*   **Value Head:** The value head of the MoE model is used to evaluate the leaf nodes of the search tree. This provides a more accurate evaluation of the positions than the traditional Monte Carlo rollout method, which can be very noisy.

The combination of MoE and MCTS allows Adela to combine the strengths of both approaches. The MoE model provides a powerful and flexible evaluation function, while MCTS provides a robust and efficient search algorithm.

## 3. Self-Training Pipeline

Adela is trained using a self-training pipeline that allows it to learn and improve by playing against itself. This process is inspired by the successful approach used by AlphaZero and Leela Chess Zero. The pipeline consists of three main stages: data generation, data storage and streaming, and training.

### 3.1. Data Generation

The training data is generated through a process of self-play. The current version of the MoE model plays games against itself, with both sides using MCTS to select moves. For each position in the game, the following information is recorded:

*   **Board State (FEN):** The board position is stored as a Forsyth-Edwards Notation (FEN) string.
*   **Policy:** The MCTS search produces a visit count for each legal move from the current position. These visit counts are normalized to create a probability distribution over the moves, which serves as the policy target for the model.
*   **Value:** The outcome of the game is used as the value target for the model. The value is +1 if the current player wins, -1 if the current player loses, and 0 for a draw.

This process generates a large dataset of high-quality training examples that can be used to improve the model.

### 3.2. Data Storage and Streaming

To handle the large amount of data generated by self-play, Adela uses the Parquet file format. Parquet is a columnar storage format that is highly efficient for storing and querying large datasets. The self-play data is written to a set of Parquet files, which can be easily processed in parallel.

To avoid loading the entire dataset into memory, Adela uses a streaming dataset implementation. The `StreamingSelfPlayDataset` class reads the data from the Parquet files in a streaming fashion, allowing the model to be trained on datasets that are much larger than the available RAM.

### 3.3. Training Process

The MoE model is trained on the self-play data using a standard training loop. The model is trained to minimize a loss function that combines the policy and value losses. The policy loss is the cross-entropy between the model's policy output and the MCTS policy target. The value loss is the mean squared error between the model's value output and the game outcome.

The training process is managed by a `Trainer` class that handles the details of the training loop, such as setting the optimizer, learning rate schedule, and other hyperparameters. The script also includes features like early stopping, which monitors the validation loss and stops the training process if the model is no longer improving. This helps to prevent overfitting and to ensure that the model generalizes well to new positions.

The self-training pipeline can be run in a continuous loop, allowing the model to continuously generate data and train, leading to progressive improvement in its playing strength. This iterative process of self-play and training is the key to Adela's ability to learn and improve over time.

## 4. Experiments and Results

To evaluate the performance of Adela, we will conduct a series of experiments to benchmark it against other strong open-source chess engines and to analyze the effectiveness of its key architectural components. The experiments will be designed to answer the following questions:

*   How does Adela's playing strength compare to state-of-the-art chess engines like Stockfish and Leela Chess Zero?
*   How much does the Mixture of Experts (MoE) architecture contribute to Adela's performance?
*   How do the specialized experts contribute to the engine's decision-making process?

### 4.1. Benchmarking against Stockfish and Leela Chess Zero

We will play a series of games between Adela and the latest versions of Stockfish and Leela Chess Zero. The matches will be played using a standard time control of 60 seconds per game with a 1-second increment. The results of these matches will be used to calculate Adela's Elo rating, which will provide a quantitative measure of its playing strength.

### 4.2. Evaluating the Mixture of Experts (MoE) Architecture

To evaluate the effectiveness of the MoE architecture, we will create a version of Adela with the MoE architecture disabled. This version of the engine will use a single, monolithic model instead of the MoE model. We will then play a series of games between the MoE version of Adela and the non-MoE version. The results of this match will be used to determine the impact of the MoE architecture on the engine's performance.

### 4.3. Analyzing Expert Contributions

To understand how the specialized experts contribute to the engine's decision-making process, we will record the expert weights assigned by the gating network during a series of games. We will then analyze these weights to determine which experts are most active in different phases of the game and against different opponents. This analysis will provide insights into the model's decision-making process and the effectiveness of the specialized experts.

_[This section will be filled in with the results of the experiments once they have been completed.]_

## 5. Conclusion and Future Work

This paper has introduced Adela, a novel chess engine that combines a Mixture of Experts (MoE) architecture with a self-training pipeline. Adela's MoE model leverages specialized experts in different aspects of the game, allowing it to adapt its strategy to the specific demands of each position. The self-training pipeline enables the engine to continuously improve its performance by learning from self-play.

The key contributions of this work are the novel MoE architecture for chess, the dynamic gating mechanism for expert selection, and the self-training pipeline for continuous improvement. We believe that this approach has the potential to achieve high-level chess performance and to advance the state of the art in computer chess.

Future work will focus on the following areas:

*   **Scaling up the self-training pipeline:** We plan to scale up the self-training pipeline by using more computing resources to generate more self-play data and to train larger models.
*   **Adding new experts:** We plan to explore the use of new experts, such as experts for specific openings or for recognizing specific tactical motifs.
*   **Improving the gating network:** We plan to investigate more sophisticated gating networks, such as attention-based models, to further improve the expert selection process.
*   **Evaluating the engine against human players:** We plan to evaluate Adela's performance against strong human players to assess its real-world playing strength.

We hope that Adela will serve as a valuable platform for future research in computer chess and artificial intelligence.
