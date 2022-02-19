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
                return (v > 0) ? v : Math.pow(Math.E, v) - 1;
            }
            case HARD_ELISH -> {
                return (v < 0) ? Math.max(0, Math.min(1, (v + 1) / 2)) * (Math.pow(Math.E, v) - 1)
                        : v * Math.max(0, Math.min(1, (v + 1) / 2));
            }
            case LEAKY_RELU -> {
            }
            case RELU -> {
            }
            case SELU -> {
            }
            case STEP -> {
            }
            case SWISH -> {
            }
            case TANH -> {
            }
        }
        return 0;
    }

    @Override
    public void run() {

    }
}
