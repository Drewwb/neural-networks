import java.util.Arrays;

public class TestNN {
        public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork(2, 2, 1);

        double[][] inputs = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };

        double[][] targets = {
            {0},
            {1},
            {1},
            {0}
        };

        nn.train(inputs, targets, 10000, 0.1);

        for (double[] input : inputs) {
            double[] output = nn.forward(input);
            System.out.println(Arrays.toString(input) + " => " + Arrays.toString(output));
        }
    }
}
