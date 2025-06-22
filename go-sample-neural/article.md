# Developing a Neural Network in Golang from Scratch: From Basics to XOR Solution

In the world of machine learning, neural networks are powerful tools for solving a wide range of tasks, from image recognition to natural language processing and games. While there are many high-level libraries that simplify the creation and training of neural networks (e.g., TensorFlow, PyTorch, Keras), understanding how these networks work "under the hood" is invaluable.

This article is dedicated to creating a simple yet functional feedforward neural network in Go, using only the standard library. This approach will allow us to delve deeply into the mechanisms of forward and backward propagation, weight initialization, and training.

## Why Golang and "From Scratch"?

Go is a compiled language known for its performance, ease of concurrency, and strong typing. These qualities make it an excellent choice for low-level implementations where control over performance is crucial. Developing a neural network from scratch in Go allows you to:

- **Deeply understand algorithms**: You implement each mathematical step, which strengthens your understanding of neural network principles.

- **Learn Go**: This is excellent practice for learning matrix operations, data structures, and working with pointers in Go.

- **Control and optimization**: Full control over every aspect of the implementation allows for deep optimization when necessary.

## Alternatives: Existing ML Libraries in Go

Before diving into the "from scratch" implementation, it's worth mentioning that for more complex and "production-ready" projects, there are specialized libraries that provide pre-built and optimized components:

- **Gorgonia**: A powerful library for building computational graphs, very similar to TensorFlow or PyTorch. It allows you to define and train complex deep learning models.

- **GoMind**: A simpler library for creating and training basic neural networks.

- **GoLearn**: While primarily a general machine learning library (including clustering, classification algorithms, etc.), it also offers some capabilities for working with neural networks.

These libraries significantly speed up development, but for understanding the fundamentals, it's best to start "from scratch."

## Architecture of Our Neural Network

We will build a **Feedforward Neural Network (FFNN)**. This means that neurons in one layer are fully connected to all neurons in the next layer, and information flows in only one direction – from input to output.

### Layer Structure: `NeuralNetworkLayer`

In our implementation, we've combined the logic of linear transformation (weighted sums and biases) and activation functions into a single `NeuralNetworkLayer` structure. This aligns with the conceptual understanding of a "layer" in most modern frameworks.

```Go
// NeuralNetworkLayer represents one fully connected layer with an activation function.
type NeuralNetworkLayer struct {
	InputSize  int
	OutputSize int
	Weights    [][]float64 // Weights[output_neuron_idx][input_neuron_idx]
	Biases     []float64

	ActivationFunc func(float64) float64
	DerivativeFunc func(float64) float64

	// Temporal values ​​for backpropagation
	InputVector  []float64 // Input values ​​to layer (from previous layer)
	WeightedSums []float64 // Values ​​after linear transformation (before activation)
	OutputVector []float64 // Output values ​​after activation

	// Gradients for updating weights and biases
	WeightGradients [][]float64
	BiasGradients   []float64
	InputGradient   []float64 // Gradient passed to the previous layer
}
```

- `Weights` and `Biases`: These are the trainable parameters of the layer. `Weights` is a matrix that multiplies the input vector, and `Biases` is a vector added to the result.

- `ActivationFunc` and `DerivativeFunc`: Pointers to activation functions (e.g., ReLU, Sigmoid) and their derivatives, which are applied to the output of the linear transformation.

### Weight Initialization: He Initialization

One of the most critical aspects is proper weight initialization. It helps prevent vanishing or exploding gradient problems that can slow down or halt training. We use **He initialization** for layers with ReLU activation, while for other types of activations (e.g., Sigmoid), a standard initialization with a smaller scale is used:

```Go
// In the NewNeuralNetworkLayer function
for i := range weights {
    weights[i] = make([]float64, inputSize)
    for j := range weights[i] {
        // Initialize weights based on the activation function
        if activationName == "relu" {
            weights[i][j] = rand.NormFloat64() * math.Sqrt(2.0/float64(inputSize)) // He initialization
        } else {
            weights[i][j] = rand.NormFloat64() * math.Sqrt(1.0/float64(inputSize)) // More general approach
        }
    }
    biases[i] = 0.0
}
```

Here, `rand.NormFloat64()` generates a random number from a standard normal distribution, and the multiplier `math.Sqrt(2.0/float64(inputSize))` scales it to maintain a stable variance of activations when using ReLU. For other activations, a multiplier of `math.Sqrt(1.0/float64(inputSize))` is used.

### Activation Functions: ReLU and Sigmoid

Our implementation supports both ReLU and Sigmoid.

- ReLU ($f(x)=max(0,x)$) is widely used in hidden layers to introduce non-linearity and combat the vanishing gradient problem.

- Sigmoid ($f(x)=1/(1+e^{−x})$) squashes values into the range of 0 to 1, often used in output layers for classification tasks where probabilities are needed.

 ```Go
 // Example activation functions from tools.go
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func SigmoidDerivative(x float64) float64 {
	s := Sigmoid(x)
	return s * (1 - s)
}

func ReLU(x float64) float64 {
	return math.Max(0, x)
}

func ReLUDerivative(x float64) float64 {
	if x > 0 {
		return 1.0
	}
	return 0.0
}
 ```

 For the output layer, a linear activation (denoted as "none" in our code) is typically used to obtain raw numerical values, such as Q-values in DQN or regression outputs.

 ### Implementation Details: Forward and Backward Pass
Let's examine the key `Forward` and `Backward` methods from `NeuralNetworkLayer`, which form the core of neural network computations.

#### `Forward` Method (Forward Pass)
The `Forward` method is responsible for computing the output of a layer based on given input data.

```Go
// Forward performs a forward pass through the layer (linear part + activation).
func (item *NeuralNetworkLayer) Forward(input []float64) []float64 {
	item.Input = input // Save input for use in Backward
	// 1. Linear transformation: multiply by weights and add biases
	item.WeightedSums = MultiplyMatrixVector(item.Weights, input)
	item.WeightedSums = AddVectors(item.WeightedSums, item.Biases)

	// 2. Activation: apply the activation function to the result of the linear transformation
	item.Output = make([]float64, len(item.WeightedSums))
	for i := range item.WeightedSums {
		item.Output[i] = item.ActivationFunc(item.WeightedSums[i])
	}
	return item.Output
}
```

At each step:

- The input vector (`input`) is multiplied by the layer's weight matrix (`item.Weights`).

- The bias vector (`item.Biases`) is added to the result. This gives `WeightedSums` (or z-values).

- The activation function (`item.ActivationFunc`) is applied to each element of `WeightedSums`, yielding the final `Output` of the layer.

#### `Backward` Method (Backward Pass)
The `Backward` method computes the gradients of the loss with respect to the layer's weights, biases, and inputs. This is the foundation of the backpropagation algorithm.

```Go
// Backward performs a reverse pass through the layer.
func (item *NeuralNetworkLayer) Backward(outputGradient []float64) []float64 {
	// 1. Gradient via activation function:
	// Apply the derivative of the activation function to the saved WeightedSums
	activationGradient := make([]float64, len(item.WeightedSums))
	for i := range item.WeightedSums {
		activationGradient[i] = item.DerivativeFunc(item.WeightedSums[i])
	}
	// Combine the gradient from the next layer with the activation gradient (element-wise multiplication)
	gradientAfterActivation := ElementWiseMultiply(outputGradient, activationGradient)

	// 2. Gradient for biases: equals the gradient after activation
	item.BiasGradients = gradientAfterActivation

	// 3. Gradient for weights: computed as the outer product
	// (gradient_after_activation X InputVector)
	item.WeightGradients = OuterProduct(gradientAfterActivation, item.Input)

	// 4. Gradient for input: computed as the product of the transposed weight matrix
	// and the gradient after activation. This is the gradient passed to the previous layer.
	transposedWeights := TransposeMatrix(item.Weights)
	item.InputGradient = MultiplyMatrixVector(transposedWeights, gradientAfterActivation)

	return item.InputGradient
}
```

The main steps of `Backward` for `NeuralNetworkLayer`:

- The gradient passing through the activation function is calculated. For this, `outputGradient` (the gradient received from the next layer) is element-wise multiplied by the derivative of the activation function, computed at the `WeightedSums`.

- The gradients for `Biases` are equal to this `gradientAfterActivation`.

- The gradients for `Weights` are computed as the outer product of `gradientAfterActivation` and the layer's `Input`.

- The gradient to be passed to the previous layer (`InputGradient`) is calculated as the product of the transposed weight matrix of the layer and `gradientAfterActivation`.

### Overall Network Structure: `NeuralNetwork`
The neural network itself (`NeuralNetwork`) is a collection of these layers:

```Go
// network.go
type NeuralNetwork struct {
	Layers []*NeuralNetworkLayer
}
...
// Predict performs a forward pass to obtain network predictions.
func (item *NeuralNetwork) Predict(input []float64) []float64 {
	output := input
	for _, layer := range item.Layers {
		output = layer.Forward(output)
	}
	return output
}

// Train performs one step of training the network.
// input: input data
// targetOutput: target output data (Q-values ​​for training)
// learningRate: learning rate
func (item *NeuralNetwork) Train(input []float64, targetOutput []float64, learningRate float64) {
	// Forward pass (saving intermediate values)
	predictedOutput := item.Predict(input)

	// Calculate the gradient of the output (MSE loss derivative)
	// dLoss/dOutput = 2 * (predicted - target)
	outputGradient := make([]float64, len(predictedOutput))
	for i := range predictedOutput {
		outputGradient[i] = 2 * (predictedOutput[i] - targetOutput[i])
	}

	// Backward pass
	currentGradient := outputGradient
	for i := len(item.Layers) - 1; i >= 0; i-- {
		currentGradient = item.Layers[i].Backward(currentGradient)
	}

	// Updating weights
	for _, layer := range item.Layers {
		layer.Update(learningRate)
	}
}
```

- `Predict`: The method for the forward pass. It simply passes the input data sequentially through each layer to get the final output.

- `Train`: The method for training. It performs a forward pass, calculates the loss function gradient (MSE), then performs backpropagation by iterating through the layers in reverse order and computing gradients for each layer. Finally, it updates the weights and biases of the layers.

## Application: Solving the XOR Problem

The XOR (exclusive OR) problem is a classic test for neural networks because it is not linearly separable. A simple network without hidden layers (a linear model) cannot solve it. However, a network with one or more hidden layers can successfully tackle XOR.

### XOR Data
The XOR problem takes two binary inputs (0 or 1) and outputs 1 if the inputs are different, and 0 if they are the same.

- Inputs: `[0,0], [0,1], [1,0], [1,1]`

- Outputs: `[0], [1], [1], [0]`

### Architecture for XOR

To solve XOR, according to the provided source code (`main.go`), we use a network with 2 input neurons, 1 hidden layer with 2 neurons (and Sigmoid activation), and 1 output neuron (with linear activation):

```Go
NeuralNetwork := NewNeuralNetwork(2, []int{2}, 1, "sigmoid")
```

This minimal architecture with one hidden layer is necessary for solving the XOR problem. Using a sigmoid in the hidden layer is a traditional approach for this task.

### Training Process

Training occurs iteratively:

- We define the learningRate and the number of epochs.

- In each epoch, we iterate through all XOR examples.

- For each example (input, target), the NeuralNetwork.Train method is called, which adjusts the network's weights.

- We track the total loss for each epoch to ensure it decreases, indicating that the network is learning.

```Go
// main.go
unc main() {
	// Define XOR data
	// Inputs: [0,0], [0,1], [1,0], [1,1]
	// Outputs: [0], [1], [1], [0]
	xorInputs := [][]float64{
		{0.0, 0.0},
		{0.0, 1.0},
		{1.0, 0.0},
		{1.0, 1.0},
	}
	xorOutputs := [][]float64{
		{0.0},
		{1.0},
		{1.0},
		{0.0},
	}
	// Neural Network architecture for XOR
	// 2 inputs, 1 hidden layer with 2 neurons, 1 output, sigmoid activation
	NeuralNetwork := neural.NewNeuralNetwork(2, []int{2}, 1, "sigmoid")
	// Test the non-trained network
	fmt.Println("Testing the non-trained network:")
	for i, input := range xorInputs {
		predictedOutput := NeuralNetwork.Predict(input)
		fmt.Printf("Input: %v, Expected: %v, Predicted: %.4f\n", input, xorOutputs[i], predictedOutput)
	}

	fmt.Println("Starting XOR training...")
	learningRate := 0.01
	epochs := 20000 // Number of training epochs
	for i := range epochs {
		totalLoss := 0.0
		for j := range xorInputs {
			input := xorInputs[j]
			target := xorOutputs[j]

			// Train the network on one XOR example
			NeuralNetwork.Train(input, target, learningRate)

			// Calculate current loss for monitoring (optional, but good practice)
			predicted := NeuralNetwork.Predict(input)
			loss := 0.0
			for k := range predicted {
				diff := predicted[k] - target[k]
				loss += diff * diff // MSE
			}
			totalLoss += loss
		}

		if (i+1)%1000 == 0 {
			fmt.Printf("Epoch %d, Average Loss: %.6f\n", i+1, totalLoss/float64(len(xorInputs)))
		}
	}
	fmt.Println("XOR training finished.")
	fmt.Println("Testing the trained network:")
	// Test the trained network
	for i, input := range xorInputs {
		predictedOutput := NeuralNetwork.Predict(input)
		fmt.Printf("Input: %v, Expected: %v, Predicted: %.4f\n", input, xorOutputs[i], predictedOutput)
	}
}
```

### Console Output Example

During training, you will see the average loss gradually decrease. Note that when using sigmoid and the specified training parameters, the results might not be perfectly 0.0 or 1.0, but they will be close enough for logical classification:

```
Testing the non-trained network:
Input: [0 0], Expected: [0], Predicted: [-0.6342]
Input: [0 1], Expected: [1], Predicted: [-0.6208]
Input: [1 0], Expected: [1], Predicted: [-0.0719]
Input: [1 1], Expected: [0], Predicted: [-0.0617]
Starting XOR training...
Epoch 1000, Average Loss: 0.210077
Epoch 2000, Average Loss: 0.153441
Epoch 3000, Average Loss: 0.056236
Epoch 4000, Average Loss: 0.003917
Epoch 5000, Average Loss: 0.000094
Epoch 6000, Average Loss: 0.000002
Epoch 7000, Average Loss: 0.000000
...
Epoch 19000, Average Loss: 0.000000
Epoch 20000, Average Loss: 0.000000
XOR training finished.
Testing the trained network:
Input: [0 0], Expected: [0], Predicted: [0.0000]
Input: [0 1], Expected: [1], Predicted: [1.0000]
Input: [1 0], Expected: [1], Predicted: [1.0000]
Input: [1 1], Expected: [0], Predicted: [0.0000]
```

As the example shows, the predicted values are very close to the expected 0 or 1, indicating successful training.

If you are interested in the weights and biases before and after training, here they are:

```
// Before training
Biases 0 -  [0 0]
Weights 0 -  [[-1.5470222607067037 -0.01696838240061016] [0.24909017310045511 0.06488626663290908]]
Biases 1 -  [0]
Weights 1 -  [[-1.6581899752969296 0.389885131323495]]

// Biases and Weights after training:
Biases 0 -  [0.3944580734523731 2.1331071307836806]
Weights 0 -  [[-3.3592486686297476 -3.275219318130251] [-1.6500618206834206 -1.631091932264212]]
Biases 1 -  [-0.7323529446320001]
Weights 1 -  [[-3.3658966295356247 3.0679477637291073]]
```

## Conclusion

We've explored how to build a basic feedforward neural network in Golang from scratch, including its architecture, activation functions, and weight initialization principles. By applying this network to solve the classic XOR problem, we've confirmed the functionality of our implementation.

In the next part of this article series, we will use this architecture as a foundation for creating a more complex system – a Deep Q-Network (DQN) based agent that can learn to play Tic-Tac-Toe independently, tackling the challenge of delayed rewards. Stay tuned for updates!