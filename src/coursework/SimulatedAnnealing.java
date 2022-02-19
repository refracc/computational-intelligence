package coursework;

import model.NeuralNetwork;

/**
 * This class is an implementation of the Simulated Annealing algorithm that can be used to attempt to solve the problem of landing lunar landers.
 * By default, the {@link SimulatedAnnealing} class extends {@link NeuralNetwork} to confer to the constraints placed upon the {@link Parameters} class' {@code neuralNetworkClass} variable.
 */
public class SimulatedAnnealing extends NeuralNetwork {

    @Override
    public double activationFunction(double v) {
        return switch (Parameters.ACTIVATION) {
            case HARD_ELISH -> (v < 0) ? Math.max(0, Math.min(1, (v + 1) / 2)) * (Math.exp(v) - 1) : v * Math.max(0, Math.min(1, (v + 1) / 2));
            case LEAKY_RELU -> (v > 0) ? v : (v / 100);
            case RELU -> (v > 0) ? v : -1;
            case SELU -> (v > 0) ? v * 1.0507009 : 1.0507009 * (1.673263 * Math.exp(v)) - 1.673263;
            case STEP -> (v <= 0) ? -1.0d : 1.0d;
            case SWISH -> (v * (1 / (1 + Math.exp(-v))));
            case TANH -> (v < -20.0d) ? -1.0d : (v > 20.0d) ? 1.0d : Math.tanh(v);
            default -> (v > 0) ? v : Math.exp(v) - 1;
        };
    }

    @Override
    public void run() {

    }
}
