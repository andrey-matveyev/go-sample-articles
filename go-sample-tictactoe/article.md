# TicTacToe. Go Duel. AI vs Fate.

### Neural Network vs Random Number Generator

Previous articles:
- [Part #1. Developing a Neural Network in Golang.](https://dev.to/andrey_matveyev/developing-a-neural-network-in-golang-bl8)

---

> "Knowledge itself is power" (с) -Francis Bacon

Creating a network is easy. Training it correctly is not an easy task. The result often does not match expectations. In reality, there is no magic here. The network does exactly what it is told to do. If the result is not what was intended, then the error is either in the training or in the interpretation of the results obtained. The creator's thoughts cannot yet be guessed by the network.

In our previous article, we delved into the fundamentals of neural networks, building a simple model in Golang and successfully solving the classic XOR problem. Now it's time to move on to a more exciting and complex area — Reinforcement Learning — and apply this knowledge to create an intelligent agent capable of playing Tic-Tac-Toe.

Unlike the XOR problem, where the network immediately received the "correct answer" and could adjust its weights, in games like Tic-Tac-Toe, a key difficulty arises: delayed reward. The agent makes moves, but the outcome of its actions (win, loss, or draw) is only known at the end of the game. This means we cannot immediately point out an "error" or "success" for each individual move to the network. The agent needs to learn to associate intermediate actions with future outcomes.

It is precisely to solve such problems that the Deep Q-Learning (DQN) algorithm was developed, which we will discuss in detail in this article. We will describe the game logic, the DQN agent's architecture, and analyze its training process as both the first and second player. The article is written in an accessible, popular style and will not delve deeply into the mathematical foundations, as there are many excellent resources on this topic available online (e.g., [mathematics of reinforcement learning (RL)](https://www.anyscale.com/blog?author=misha-laskin) or [video about DeepLearning](https://www.youtube.com/playlist?list=PLZjXXN70PH5itkSPe6LTS-yPyl5soOovc)).

### Tic-Tac-Toe Game Logic

Tic-Tac-Toe is a simple deterministic game for two players on a 3x3 board. Players take turns placing their symbols (X and O) into empty cells. The goal of the game is to be the first to get three of your symbols in a row horizontally, vertically, or diagonally. If all cells are filled and no winner is determined, the game ends in a draw.

**Key aspects of the game:**

- **Game State**: Determined by the arrangement of X and O symbols on the board.

- **Actions**: Choosing an empty cell to place your symbol.

- **Game Outcome**: A win for one of the players or a draw.

- **First Move Advantage**: In Tic-Tac-Toe, the first player has a strategic advantage. With optimal play from both players, the game always ends in a draw or a win for the first player. According to my estimates, and confirmed by experiment (when the agent initially plays like a random opponent), the probability of winning for the player who makes the first move to the center is about 60% (600 out of 1000 games), a loss is about 30%, and a draw is 10%.

**Board Representation and State Vector**

The game `board` is represented by a Board struct, and its state is converted into a numerical vector for the neural network using the `GetStateVector` method.

```Go
// tictactoe.go

// Board represents the Tic-Tac-Toe game board.
type Board struct {
	Cells         [9]int // 0: empty, 1: X, -1: O
	CurrentPlayer int    // 1 for X, -1 for O
}

// NewBoard creates a new empty board.
func NewBoard() *Board {
	return &Board{
		Cells:         [9]int{Empty, Empty, Empty, Empty, Empty, Empty, Empty, Empty, Empty},
		CurrentPlayer: PlayerX, // X always starts
	}
}

// GetStateVector converts the board state into a vector for the neural network.
// Represents the 3x3 board as a flat 9-element vector.
// 1.0 for agent's cell, -1.0 for opponent's cell, 0.0 for empty.
func (item *Board) GetStateVector(agentPlayer int) []float64 {
	state := make([]float64, 9)
	for i, cell := range item.Cells {
		switch cell {
		case agentPlayer:
			state[i] = 1.0
		case -agentPlayer: // Opponent
			state[i] = -1.0
		default: // Empty
			state[i] = 0.0
		}
	}
	return state
}
```

### Deep Q-Learning Agent (DQN)

Our agent is based on the **Double DQN** architecture, which combines Q-learning with deep neural networks. This is evident in how the action for the next state is selected using the main Q-network, and then its Q-value is evaluated using the target network. This helps to reduce the overestimation of Q-values characteristic of classic DQN.

**State Representation**

The board state is converted into a numerical vector that is fed into the neural network. For each of the 9 board cells:

- `1.0`, if the cell is occupied by the agent's symbol.

- `-1.0`, if the cell is occupied by the opponent's symbol.

- `0.0`, if the cell is empty.

**Neural Network Architecture**

The agent uses a fully connected neural network.

- **Input Layer**: 9 neurons (corresponding to the 9 board cells).

- **Hidden Layer**: One hidden layer with 27 (or 45/72) neurons with a Tanh activation function. The minimum number of neurons in the hidden layer that yielded satisfactory results was 9.

- **Output Layer**: 9 neurons (corresponding to 9 possible actions/cells), also with a Tanh activation function.

### DQN Training Mechanism

The agent learns by interacting with the environment (the Tic-Tac-Toe game) and receiving rewards.

**Experience Replay Buffer**

The `ReplayBuffer` stores the agent's experiences, allowing for efficient training by sampling past interactions.

```Go
// agent.go

// Experience represents a single game experience.
type Experience struct {
	State     []float64
	Action    int
	Reward    float64
	NextState []float64
	Done      bool
}

// ReplayBuffer stores game experiences.
type ReplayBuffer struct {
	Experiences []Experience
	Capacity    int

	Index int
	Size  int
}

// NewReplayBuffer creates a new experience buffer.
func NewReplayBuffer(capacity int) *ReplayBuffer {
	return &ReplayBuffer{
		Experiences: make([]Experience, capacity),
		Capacity:    capacity,
	}
}

// Add adds a new experience to the buffer.
func (item *ReplayBuffer) Add(exp Experience) {
	item.Experiences[item.Index] = exp
	item.Index = (item.Index + 1) % item.Capacity
	if item.Size < item.Capacity {
		item.Size++
	}
}

// Sample selects a random batch of experiences from the buffer.
func (item *ReplayBuffer) Sample(batchSize int) []Experience {
	if item.Size < batchSize {
		return nil // Not enough experience to sample a batch
	}

	samples := make([]Experience, batchSize)
	for i := range batchSize {
		idx := rand.Intn(item.Size)
		samples[i] = item.Experiences[idx]
	}
	return samples
}
```

**DQNAgent Structure and Action Selection**

The `DQNAgent` struct holds the Q-network, target network, replay buffer, and training parameters. The `ChooseAction` method implements the epsilon-greedy strategy.

```Go
// DQNAgent represents a Deep Q-Learning agent.
type DQNAgent struct {
	QNetwork      *NeuralNetwork
	TargetNetwork *NeuralNetwork
	ReplayBuffer  *ReplayBuffer
	Gamma         float64 // Discount factor
	MaxEpsilon    float64 // For epsilon-greedy strategy
	MinEpsilon    float64 // Minimum epsilon value
	EpsilonDecay  float64 // Epsilon decay rate per episode
	LearningRate  float64
	UpdateTarget  int // Target network update interval (steps)
	PlayerSymbol  int // Symbol this agent plays (PlayerX or PlayerO)
}

// NewDQNAgent creates a new DQN agent.
func NewDQNAgent(inputSize, outputSize, bufferCapacity int, playerSymbol int) *DQNAgent {
	qNet := NewNeuralNetwork(inputSize, []int{hiddenLayerSize}, outputSize, "tanh") // Example architecture
	targetNet := qNet.Clone()                                                       // Clone for the target network

	return &DQNAgent{
		QNetwork:      qNet,
		TargetNetwork: targetNet,
		ReplayBuffer:  NewReplayBuffer(bufferCapacity),
		Gamma:         gamma,
		MaxEpsilon:    maxEpsilon,
		MinEpsilon:    minEpsilon,
		EpsilonDecay:  epsilonDecay,
		LearningRate:  learningRate,
		UpdateTarget:  updateTarget,
		PlayerSymbol:  playerSymbol,
	}
}

// ChooseAction selects an action using an epsilon-greedy strategy.
// board: current board state.
func (item *DQNAgent) ChooseAction(board *Board) int {
	emptyCells := board.GetEmptyCells()

	// Epsilon-greedy strategy: random move or best move according to Q-network
	if rand.Float64() < item.MaxEpsilon {
		return emptyCells[rand.Intn(len(emptyCells))] // Random move (Research process)
	}

	// Choose the best move according to the Q-network
	stateVec := board.GetStateVector(item.PlayerSymbol)
	qValues := item.QNetwork.Predict(stateVec)

	bestAction := -1
	maxQ := -math.MaxFloat64 // Initialize with a very small number
	for _, action := range emptyCells { // Iterate ONLY through empty cells
		if qValues[action] > maxQ {
			maxQ = qValues[action]
			bestAction = action // Found a new maximum
		}
	}
	return bestAction
}
```
**Training the Agent**

The `Train` method implements the core Double DQN update rule, using the replay buffer and target network.

```Go
// Train performs one training step for the agent.
// batchSize: batch size for training.
// step: current step (for target network update).
func (item *DQNAgent) Train(batchSize, step int) {
	batch := item.ReplayBuffer.Sample(batchSize)
	if batch == nil {
		return // Not enough experience
	}

	for _, exp := range batch {
		// Predicted Q-values for the current state from the Q-network
		currentQValues := item.QNetwork.Predict(exp.State)
		targetQValues := make([]float64, len(currentQValues))
		copy(targetQValues, currentQValues) // Copy to modify only one value

		// Calculate the target Q-value
		var targetQ float64
		if exp.Done {
			targetQ = exp.Reward // If the game is over, the target value is the immediate reward
		} else {
			// 1. Get Q-values for the next state from the Q-network (to choose the best action)
			qValuesNextStateFromQNetwork := item.QNetwork.Predict(exp.NextState)
			// Find the action that would be chosen by the Q-network in the next state.
			bestActionFromQNetwork := -1
			maxQValFromQNetwork := -math.MaxFloat64
			// Find the index of the best action from the Q-network's predictions.
			for i, qVal := range qValuesNextStateFromQNetwork {
				if qVal > maxQValFromQNetwork {
					maxQValFromQNetwork = qVal
					bestActionFromQNetwork = i
				}
			}
			// 2. Evaluate the Q-value of the chosen action using the Target Network
			qValueFromTargetNetwork := item.TargetNetwork.Predict(exp.NextState)[bestActionFromQNetwork]
			targetQ = exp.Reward + item.Gamma*qValueFromTargetNetwork // Bellman Equation (DDQN) !!!
		}
		// Update the target Q-value for the action taken in this experience
		targetQValues[exp.Action] = targetQ
		// Train the Q-network with the updated target Q-values
		item.QNetwork.Train(exp.State, targetQValues, item.LearningRate)
	}
	// Decay epsilon (applied per training step, not per episode)
	if item.MaxEpsilon > item.MinEpsilon {
		item.MaxEpsilon *= item.EpsilonDecay
	}
	// Update the target network
	if step%item.UpdateTarget == 0 {
		item.TargetNetwork = item.QNetwork.Clone()
		fmt.Printf("--- Target network updated at step %d (Epsilon: %.4f) ---\n", step, item.MaxEpsilon)
	}
}
```

Note the Bellman equation:

```Go
  ...
  targetQ = exp.Reward + item.Gamma*qValueFromTargetNetwork
  ...
```

Using this mechanism, the "reward" gradually "propagates" from the end of the game to its beginning.

**Rewards**

The `GetReward` function defines the reward structure for the agent:

```Go
// tictactoe.go

// GetReward returns the reward for the agent based on the game outcome.
// This function is called AFTER a move has been made and the game state potentially changed.
func (item *Board) GetReward(agentPlayer int) float64 {
	isOver, winner := item.GetGameOutcome()
	if isOver {
		switch winner {
		case agentPlayer:
			return winsReward // Agent wins
		case Empty:
			return drawReward // Draw
		default: // winner == -agentPlayer (opponent)
			return losesReward // Agent loses (opponent wins)
		}
	}
	return 0.0 // No negative reward for moves in Tic-Tac-Toe
}
```

### Training Process and Results

**Training the Agent as the First Player**

In this scenario, the agent (Player X) always makes the first move in the game. To accelerate convergence and ensure the learning of an optimal starting strategy, we can experimented with forcing the first move to the center of the board (default without this).

**Training Parameters**:

```Go
// main.go
const (
	// Who makes the first move (first step)?
	agentsFirstStep bool = true // true = agent (PlayerX), false = opponent (PlayerO)
	// Training parameters
	episodes       int = 500000 // Number of game episodes for training
	batchSize      int = 8      // Batch size for DQN training
	bufferCapacity int = 50000  // Experience buffer capacity
	trainStartSize int = 1000   // Start training after accumulating enough experience
	// Learning parameters for DQNAgent
	gamma        float64 = 0.75     // Discount factor (how much the agent values future rewards)
	maxEpsilon   float64 = 1.0      // Start with exploration
	minEpsilon   float64 = 0.001    // Minimum epsilon value
	epsilonDecay float64 = 0.999996 // Epsilon decay rate per step (very slow)
	learningRate float64 = 0.0002   //
	updateTarget int     = 50000    // Update target network every 10000 steps (less frequently)
	// Reward parameters
	winsReward  float64 = 0.999
	drawReward  float64 = 0.001
	losesReward float64 = -1.000
	// Hidden layer size
	hiddenLayerSize int = 27
)
```

**Results:**

When the agent was to make the first move to the center, it demonstrated outstanding results, achieving up to 992 wins out of 1000 test games against a random opponent, with a minimal number of losses and draws. This confirms that the agent successfully learned an optimal strategy for the first player.

**Training the Agent as the Second Player**

In this scenario, the opponent (Player O) always makes the first move randomly, and our agent (Player X) always responds second. This puts the agent in a less advantageous position, as the first move in Tic-Tac-Toe provides a strategic advantage. The goal of this experiment is to test how well the agent can adapt to the role of the second player and minimize the opponent's advantage.

**Training Parameters**:

- The same hyperparameters as for the first scenario were used.

- The only change is that the opponent always makes the first move.

```Go
// main.go
const (
	// Who makes the first move (first step)?
	agentsFirstStep bool = false // true = agent (PlayerX), false = opponent (PlayerO)
    ...
)
```

**Results:**

In the initial stages of training, the agent, as expected, showed a lower win percentage and a higher number of losses/draws due to the opponent's first-move advantage. However, as training progressed, the agent significantly improved its performance.

**Example results after training:**

These results show that the agent successfully learned to optimally respond to various first moves by the opponent, significantly increasing its win rate despite the strategic disadvantage of moving second. The "first move selection" problem for the agent disappeared, as it focused on reactive tactics.

**Agent competence growth chart:**

### Conclusion

The project on training a DQN agent for Tic-Tac-Toe successfully demonstrated the effectiveness of deep reinforcement learning algorithms even for simple deterministic games. We saw how the agent can adapt to different roles (first/second player) and achieve near-optimal performance against a random opponent.


#### Postscript

The most guaranteed way to make the agent learn "human" optimality (center, corners) is to train it against a stronger, strategic opponent (e.g., Minimax AI) or in self-play mode. These opponents will punish any suboptimal move, forcing the agent towards true optimality.

Write in the comments if you are interested, and I will arrange a battle (a real fight) between two agents. For now, my immediate plans include a final "move" to Linux and writing a small backend (e.g., a REST API) for a simple client to try playing with what has been developed.