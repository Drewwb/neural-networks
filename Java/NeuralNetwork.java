/*
 * Feedforward Neural Network (Java)
 * 
 * A Feedforward Neural Network (FNN) is the simplest type of artificial neural network. It consists of 
 * layers of nodes (neurons) where each layer is fully connected to the next one, and the data moves in 
 * one direction, from input to output. This type of network is commonly used for tasks such as 
 * regression and classification.
 * 
 * -- Drew Brown
 */

public class NeuralNetwork {
    private int inputSize;
    private int hiddenSize;
    private int outputSize;
    private double[][] weightsInputHidden;
    private double[][] weightsHiddenOutput;
    private double[] hiddenBiases;
    private double[] outputBiases;

    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        weightsInputHidden = new double[inputSize][hiddenSize];
        weightsHiddenOutput = new double[hiddenSize][outputSize];
        hiddenBiases = new double[hiddenSize];
        outputBiases = new double[outputSize];
        initializeWeights();
    }

    private void initializeWeights() {
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weightsInputHidden[i][j] = Math.random() - 0.5;
            }
        }
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weightsHiddenOutput[i][j] = Math.random() - 0.5;
            }
            hiddenBiases[i] = Math.random() - 0.5;
        }
        for (int i = 0; i < outputSize; i++) {
            outputBiases[i] = Math.random() - 0.5;
        }
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    private double sigmoidDerivative(double x) {
        return x * (1 - x);
    }

    public double[] forward(double[] input) {
        double[] hiddenLayer = new double[hiddenSize];
        double[] outputLayer = new double[outputSize];

        // Input to hidden layer
        for (int i = 0; i < hiddenSize; i++) {
            hiddenLayer[i] = 0;
            for (int j = 0; j < inputSize; j++) {
                hiddenLayer[i] += input[j] * weightsInputHidden[j][i];
            }
            hiddenLayer[i] += hiddenBiases[i];
            hiddenLayer[i] = sigmoid(hiddenLayer[i]);
        }

        // Hidden to output layer
        for (int i = 0; i < outputSize; i++) {
            outputLayer[i] = 0;
            for (int j = 0; j < hiddenSize; j++) {
                outputLayer[i] += hiddenLayer[j] * weightsHiddenOutput[j][i];
            }
            outputLayer[i] += outputBiases[i];
            outputLayer[i] = sigmoid(outputLayer[i]);
        }

        return outputLayer;
    }

    public void train(double[][] inputs, double[][] targets, int epochs, double learningRate) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                double[] input = inputs[i];
                double[] target = targets[i];

                // Forward pass
                double[] hiddenLayer = new double[hiddenSize];
                double[] outputLayer = new double[outputSize];

                for (int j = 0; j < hiddenSize; j++) {
                    hiddenLayer[j] = 0;
                    for (int k = 0; k < inputSize; k++) {
                        hiddenLayer[j] += input[k] * weightsInputHidden[k][j];
                    }
                    hiddenLayer[j] += hiddenBiases[j];
                    hiddenLayer[j] = sigmoid(hiddenLayer[j]);
                }

                for (int j = 0; j < outputSize; j++) {
                    outputLayer[j] = 0;
                    for (int k = 0; k < hiddenSize; k++) {
                        outputLayer[j] += hiddenLayer[k] * weightsHiddenOutput[k][j];
                    }
                    outputLayer[j] += outputBiases[j];
                    outputLayer[j] = sigmoid(outputLayer[j]);
                }

                // Calculate errors
                double[] outputErrors = new double[outputSize];
                for (int j = 0; j < outputSize; j++) {
                    outputErrors[j] = target[j] - outputLayer[j];
                }

                double[] hiddenErrors = new double[hiddenSize];
                for (int j = 0; j < hiddenSize; j++) {
                    hiddenErrors[j] = 0;
                    for (int k = 0; k < outputSize; k++) {
                        hiddenErrors[j] += outputErrors[k] * weightsHiddenOutput[j][k];
                    }
                    hiddenErrors[j] *= sigmoidDerivative(hiddenLayer[j]);
                }

                // Backpropagation
                for (int j = 0; j < hiddenSize; j++) {
                    for (int k = 0; k < outputSize; k++) {
                        weightsHiddenOutput[j][k] += learningRate * outputErrors[k] * hiddenLayer[j];
                    }
                }

                for (int j = 0; j < inputSize; j++) {
                    for (int k = 0; k < hiddenSize; k++) {
                        weightsInputHidden[j][k] += learningRate * hiddenErrors[k] * input[j];
                    }
                }

                for (int j = 0; j < hiddenSize; j++) {
                    hiddenBiases[j] += learningRate * hiddenErrors[j];
                }

                for (int j = 0; j < outputSize; j++) {
                    outputBiases[j] += learningRate * outputErrors[j];
                }
            }
        }
    }
}