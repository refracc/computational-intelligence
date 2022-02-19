package coursework;

import model.NeuralNetwork;

/**
 * This class is an implementation of the Simulated Annealing algorithm that can be used to attempt to solve the problem of landing lunar landers.
 * By default, the {@link SimulatedAnnealing} class extends {@link NeuralNetwork} to confer to the constraints placed upon the {@link Parameters} class' {@code neuralNetworkClass} variable.
 */
public class SimulatedAnnealing extends NeuralNetwork {

    @Override
    public double activationFunction(double v) {
        switch (Parameters.ACTIVATION) {
            default -> {
                return (v > 0) ? v : Math.exp(v) - 1;
            }
            case HARD_ELISH -> {
                return (v < 0) ? Math.max(0, Math.min(1, (v + 1) / 2)) * (Math.exp(v) - 1)
                        : v * Math.max(0, Math.min(1, (v + 1) / 2));
            }
            case LEAKY_RELU -> {
                return (v > 0) ? v : (v / 100);
            }
            case RELU -> {
                return (v > 0) ? v : -1;
            }
            case SELU -> {
                return (v > 0) ? v * 1.0507009
                        : 1.0507009 * (1.673263 * Math.exp(v)) - 1.673263;
            }
            case STEP -> {
                return (v <= 0) ? -1.0d : 1.0d;
            }
            case SWISH -> {
                return (v * (1 / (1 + Math.exp(-v))));
            }
            case TANH -> {
                return (v < -20.0d) ? -1.0d : (v > 20.0d) ? 1.0d : Math.tanh(v);
            }
        }
    }

    @Override
    public void run() {

    }
}
