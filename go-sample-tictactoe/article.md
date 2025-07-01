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

So, we are all set for testing.
Let's briefly summarize what we have:
- A **neural network** with a 9:27:9 architecture that knows nothing.
- A **game board** and implementation of game logic (start, rule adherence, and end detector (win/loss/draw)).
- An **opponent** who can make moves into free cells randomly. And that's all.
- An **agent** that, from the start, plays like its opponent but has the ability to learn. It knows when the game ends. And it knows whether it finished the game well or poorly.

What can we observe and by what criteria can we determine the learning progress?
- Firstly, it's the agent's win percentage (expected to increase).
- Secondly, we can observe the decrease in Epsilon to understand what is happening – whether the agent is exploring (making random moves) or utilizing its accumulated experience.
- Thirdly, we can look at the weight vector on the output layer to understand how the agent decides to make its first move on an empty board (it is expected that the center will have the largest weight, then the corners, and then the sides as the least promising).
- And finally, we can track the maximum number of wins achieved throughout the entire experiment.

Let's see what came of this and whether our agent will show growth in its competence.

**Training the Agent as the First Player**

In this scenario, the agent (Player X) always makes the first move in the game. To accelerate convergence and ensure the learning of an optimal starting strategy, we can experimented with forcing the first move to the center of the board (default without this).

**Training Parameters**:

These are the settings that can be changed when conducting an experiment.
'Knobs' that can be 'turned' for fine-tuning.
The network implemented here usually forgives even gross errors.
The most you risk is falling into a local minimum instead of a global one.
Feel free to try it yourself.

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

```
PS D:\go\go-sample-tictactoe> go run .
Starting DQN agent training (X) against a random opponent (O) for Tic-Tac-Toe...
Episode: 1000, Wins X: 571 (571), Losses X: 307, Draws: 122, Epsilon X: 0.9876, Q(start): 0.4501|0.5164|0.4117  0.5863[0.5449]0.4485  0.3473|0.4411|0.4166
Episode: 2000, Wins X: 590 (590), Losses X: 284, Draws: 126, Epsilon X: 0.9715, Q(start): 0.3683|0.4917|0.3963  0.2354[0.6179]0.3571  0.2806|0.3732|0.3737
Episode: 3000, Wins X: 585 (590), Losses X: 294, Draws: 121, Epsilon X: 0.9558, Q(start): 0.2797|0.4310|0.3559  0.1067[0.4802]0.2719  0.1742|0.2720|0.2669
Episode: 4000, Wins X: 588 (590), Losses X: 285, Draws: 127, Epsilon X: 0.9402, Q(start): 0.2361|0.4065|0.3263  0.1037[0.3945]0.2356  0.1445|0.2771|0.2186
...
Episode: 297000, Wins X: 952 (969), Losses X: 43, Draws: 5, Epsilon X: 0.0156, Q(start): 0.5193|0.3906|0.2095  0.5050[0.3286]0.4332  0.1040|0.3630|0.2807
Episode: 298000, Wins X: 957 (969), Losses X: 40, Draws: 3, Epsilon X: 0.0154, Q(start): 0.5189|0.3942|0.1822  0.4883[0.3528]0.4347  0.1214|0.3698|0.2528
Episode: 299000, Wins X: 977 (977), Losses X: 20, Draws: 3, Epsilon X: 0.0152, Q(start): 0.5201|0.4159|0.1651  0.4708[0.3775]0.4352  0.1291|0.3870|0.2078
--- Target network updated at step 1050000 (Epsilon: 0.0151) ---
Episode: 300000, Wins X: 968 (977), Losses X: 23, Draws: 9, Epsilon X: 0.0150, Q(start): 0.4733|0.4222|0.1718  0.4519[0.4072]0.4743  0.1526|0.4102|0.1889
...
Episode: 497000, Wins X: 952 (990), Losses X: 43, Draws: 5, Epsilon X: 0.0011, Q(start): 0.3910|-0.3152|-0.2335  -0.2994[0.4932]0.0485  0.0135|-0.4090|-0.2174
--- Target network updated at step 1700000 (Epsilon: 0.0011) ---
Episode: 498000, Wins X: 942 (990), Losses X: 55, Draws: 3, Epsilon X: 0.0011, Q(start): 0.3798|-0.3127|-0.2245  -0.3118[0.4557]0.0439  0.0072|-0.4120|-0.2115
Episode: 499000, Wins X: 936 (990), Losses X: 56, Draws: 8, Epsilon X: 0.0011, Q(start): 0.3651|-0.3107|-0.2292  -0.3250[0.3711]0.0254  -0.0033|-0.4216|-0.1881
Episode: 500000, Wins X: 954 (990), Losses X: 41, Draws: 5, Epsilon X: 0.0011, Q(start): 0.3561|-0.3119|-0.2014  -0.3267[0.3711]0.0196  -0.0191|-0.4155|-0.1827

Training complete.
Testing the agent (X against random O)...

Test Results (1000 games, Agent X vs random O):
Agent X Wins: 956
Agent X Losses (Random O Wins): 39
Draws: 5
```

When the agent was to make the first move to the center, it demonstrated outstanding results, achieving up to 992 wins out of 1000 (in some cases) test games against a random opponent, with a minimal number of losses and draws. This confirms that the agent successfully learned an optimal strategy for the first player.

"Win Growth (agent moves first)" graph:

![Win Growth (agent moves first)" graph](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/83dbqlau2bi6czx9picr.png)

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

```
PS D:\go\go-sample-tictactoe> go run .
Starting DQN agent training (X) against a random opponent (O) for Tic-Tac-Toe...
Episode: 1000, Wins X: 296 (296), Losses X: 587, Draws: 117, Epsilon X: 0.9902, Q(start): 0.2536|0.3091|0.2323  0.3227[0.3963]0.3577  0.4702|0.4281|0.2465
Episode: 2000, Wins X: 298 (298), Losses X: 590, Draws: 112, Epsilon X: 0.9766, Q(start): 0.1909|0.3386|0.2124  0.3879[0.3856]0.3629  0.5409|0.4653|0.2537
Episode: 3000, Wins X: 295 (298), Losses X: 598, Draws: 107, Epsilon X: 0.9633, Q(start): 0.0990|0.3089|0.1477  0.3343[0.3218]0.2929  0.5055|0.4229|0.2093
Episode: 4000, Wins X: 261 (298), Losses X: 601, Draws: 138, Epsilon X: 0.9501, Q(start): 0.0718|0.2712|0.0945  0.3015[0.2998]0.2637  0.4218|0.3067|0.1649
...
Episode: 69000, Wins X: 610 (610), Losses X: 342, Draws: 48, Epsilon X: 0.3986, Q(start): 0.5987|0.5451|0.5798  0.5912[0.6872]0.5793  0.6331|0.5710|0.5508
Episode: 70000, Wins X: 610 (610), Losses X: 359, Draws: 31, Epsilon X: 0.3935, Q(start): 0.5962|0.5428|0.5695  0.5917[0.6848]0.5758  0.6282|0.5837|0.5531
Episode: 71000, Wins X: 606 (610), Losses X: 365, Draws: 29, Epsilon X: 0.3885, Q(start): 0.5914|0.5330|0.5650  0.5899[0.6844]0.5742  0.6268|0.5863|0.5423
Episode: 72000, Wins X: 570 (610), Losses X: 407, Draws: 23, Epsilon X: 0.3835, Q(start): 0.5867|0.5349|0.5650  0.5872[0.6871]0.5795  0.6202|0.5833|0.5385
Episode: 73000, Wins X: 564 (610), Losses X: 405, Draws: 31, Epsilon X: 0.3786, Q(start): 0.5912|0.5303|0.5606  0.5833[0.6815]0.5811  0.6198|0.5832|0.5418
Episode: 74000, Wins X: 612 (612), Losses X: 353, Draws: 35, Epsilon X: 0.3737, Q(start): 0.5958|0.5287|0.5575  0.5840[0.6816]0.5730  0.6146|0.5765|0.5359
--- Target network updated at step 250000 (Epsilon: 0.3694) ---
Episode: 75000, Wins X: 588 (612), Losses X: 373, Draws: 39, Epsilon X: 0.3689, Q(start): 0.6005|0.5305|0.5658  0.5903[0.6910]0.5730  0.6132|0.5845|0.5456
Episode: 76000, Wins X: 650 (650), Losses X: 311, Draws: 39, Epsilon X: 0.3642, Q(start): 0.6314|0.5703|0.5932  0.6218[0.7187]0.6036  0.6409|0.6085|0.5756
...
Episode: 497000, Wins X: 792 (822), Losses X: 185, Draws: 23, Epsilon X: 0.0020, Q(start): 0.5345|0.3504|0.2066  0.2787[0.5258]0.4991  0.1034|0.5461|0.5410
Episode: 498000, Wins X: 804 (822), Losses X: 168, Draws: 28, Epsilon X: 0.0020, Q(start): 0.5329|0.3472|0.2169  0.2769[0.5331]0.4969  0.1012|0.5451|0.5428
Episode: 499000, Wins X: 782 (822), Losses X: 180, Draws: 38, Epsilon X: 0.0019, Q(start): 0.5315|0.3456|0.2200  0.2724[0.5288]0.4962  0.1074|0.5430|0.5417
Episode: 500000, Wins X: 780 (822), Losses X: 188, Draws: 32, Epsilon X: 0.0019, Q(start): 0.5310|0.3443|0.2219  0.2718[0.5285]0.4971  0.1044|0.5442|0.5446

Training complete.
Testing the agent (X against random O)...

Test Results (1000 games, Agent X vs random O):
Agent X Wins: 783
Agent X Losses (Random O Wins): 191
Draws: 26
```

In the initial stages of training, the agent, as expected, showed a lower win percentage and a higher number of losses/draws due to the opponent's first-move advantage. However, as training progressed, the agent significantly improved its performance.

**Example game after training:**

```
Example game after training (X vs random O):
-------------
|   |   |   |
-------------
|   |   |   |
-------------
| O |   |   |
-------------
X's Turn:
-------------
|   |   |   |
-------------
|   |   | X |
-------------
| O |   |   |
-------------
O's Turn:
-------------
|   |   |   |
-------------
| O |   | X |
-------------
| O |   |   |
-------------
X's Turn:
-------------
|   |   |   |
-------------
| O |   | X |
-------------
| O |   | X |
-------------
O's Turn:
-------------
|   |   |   |
-------------
| O |   | X |
-------------
| O | O | X |
-------------
X's Turn:
-------------
|   |   | X |
-------------
| O |   | X |
-------------
| O | O | X |
-------------
Game Over! Player X won!
```

"Win Growth (agent moves second)" graph:

![Win Growth (agent moves second)](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/cth1rd7mu9nfgjfid8kb.png)

These results show that the agent successfully learned to optimally respond to various first moves by the opponent, significantly increasing its win rate despite the strategic disadvantage of moving second. The "first move selection" problem for the agent disappeared, as it focused on reactive tactics.

### Conclusion

The project on training a DQN agent for Tic-Tac-Toe successfully demonstrated the effectiveness of deep reinforcement learning algorithms even for simple deterministic games. We saw how the agent can adapt to different roles (first/second player) and achieve near-optimal performance against a random opponent.

The full source code is available at the link:
[https://github.com/andrey-matveyev/go-sample-tictactoe](https://github.com/andrey-matveyev/go-sample-tictactoe)

#### Postscript

The most guaranteed way to make the agent learn "human" optimality (center, corners) is to train it against a stronger, strategic opponent (e.g., Minimax AI) or in self-play mode. These opponents will punish any suboptimal move, forcing the agent towards true optimality.

Write in the comments if you are interested, and I will arrange a battle (a real fight) between two agents. For now, my immediate plans include a final "move" to Linux and writing a small backend (e.g., a REST API) for a simple client to try playing with what has been developed.