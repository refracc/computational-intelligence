package coursework;

import model.Fitness;
import model.Individual;
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
            default -> (v > 0) ? v : (Math.exp(v) - 1) / 10;
        };
    }

    @Override
    public void run() {
        double temperature = Parameters.TEMPERATURE;
        double cooling = Parameters.COOLING_RATE;

        // Initialise a new individual and
        // evaluate its fitness.
        Individual individual = new Individual();
        individual.fitness = Fitness.evaluate(individual, this);

        // Set this new individual as
        // the best individual.
        best = individual.copy();

        // Do for all evaluations in the program.
        for (int i = 0; i < Parameters.maxEvaluations; i++) {
            Individual newIndividual = individual.copy();

            // Obtain pseudo-random gene locations from within the chromosome.
            int position1 = Parameters.random.nextInt(newIndividual.chromosome.length);
            int position2 = Parameters.random.nextInt(newIndividual.chromosome.length);

            // Set up gene swaps
            double swap1 = newIndividual.chromosome[position1];
            double swap2 = newIndividual.chromosome[position2];

            // Perform the swap.
            newIndividual.chromosome[position1] = swap2;
            newIndividual.chromosome[position2] = swap1;

            // Evaluate its fitness.
            newIndividual.fitness = Fitness.evaluate(newIndividual, this);

            // Decide to accept neighbouring chromosome.
            double acceptance = acceptance(individual.fitness, newIndividual.fitness, temperature);
            individual = (acceptance > Parameters.random.nextDouble()) ? newIndividual.copy() : individual;

            // Replace the best solution that has been found thus far...
            best = (individual.fitness < best.fitness) ? individual.copy() : best;

            // Cool the system down a little...
            temperature *= (1 - cooling);
            System.out.println(i + "\t" + best);
            outputStats();
        }
        saveNeuralNetwork();
    }

    private double acceptance(double currentFitness, double newFitness, double temperature) {
        return (newFitness < currentFitness) ? 1.0d : Math.exp((currentFitness - newFitness) / temperature);
    }
}
