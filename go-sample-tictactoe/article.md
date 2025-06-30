# TicTacToe. Go Duel. AI vs Fate.

### Neural Network vs Random Number Generator

Previous articles:
- [Part #1. Developing a Neural Network in Golang.](https://dev.to/andrey_matveyev/developing-a-neural-network-in-golang-bl8)

---

> "Knowledge itself is power" — Francis Bacon

Creating a network is easy. Training it correctly is not an easy task. The result often does not match expectations. In reality, there is no magic here. The network does exactly what it is told to do. If the result is not what was intended, then the error is either in the training or in the interpretation of the results obtained. The creator's thoughts cannot yet be guessed by the network.

In our previous article, we delved into the fundamentals of neural networks, building a simple model in Golang and successfully solving the classic XOR problem. Now it's time to move on to a more exciting and complex area — Reinforcement Learning — and apply this knowledge to create an intelligent agent capable of playing Tic-Tac-Toe.

Unlike the XOR problem, where the network immediately received the "correct answer" and could adjust its weights, in games like Tic-Tac-Toe, a key difficulty arises: delayed reward. The agent makes moves, but the outcome of its actions (win, loss, or draw) is only known at the end of the game. This means we cannot immediately point out an "error" or "success" for each individual move to the network. The agent needs to learn to associate intermediate actions with future outcomes.

It is precisely to solve such problems that the Deep Q-Learning (DQN) algorithm was developed, which we will discuss in detail in this article. We will describe the game logic, the DQN agent's architecture, and analyze its training process as both the first and second player. The article is written in an accessible, popular style and will not delve deeply into the mathematical foundations, as there are many excellent resources on this topic available online (e.g., [mathematics of reinforcement learning (RL)](https://www.anyscale.com/blog?author=misha-laskin) or [video about DeepLearning](https://www.youtube.com/playlist?list=PLZjXXN70PH5itkSPe6LTS-yPyl5soOovc)).
