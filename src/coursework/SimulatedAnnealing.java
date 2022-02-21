package coursework;

import coursework.utility.Helpers;
import model.Fitness;
import model.Individual;
import model.NeuralNetwork;

/**
 * This class is an implementation of the Simulated Annealing algorithm that can be used to attempt to solve the problem of landing lunar landers.
 * By default, the {@link SimulatedAnnealing} class extends {@link NeuralNetwork} to confer to the constraints placed upon the {@link Parameters} class' {@code neuralNetworkClass} variable.
 *
 * @author Stewart A
 * @version 1.26
 * @since 19/02/2022
 */
public class SimulatedAnnealing extends NeuralNetwork {

    /**
     * The function that will be setting up the Activation Function for the Neural Network
     *
     * @param v A {@link Double} value that is given by the {@link NeuralNetwork}.
     * @return Some value based on the Activation Type set out in the {@link Parameters} class.
     */
    @Override
    public double activationFunction(double v) {
        return Helpers.activationFunction(v);
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

            // Cool the algorithm down a little...
            temperature *= (1 - cooling);
            System.out.println(i + "\t" + best);

            // Write the statistics to the screen
            outputStats();
        }
        // Save the Neural Network weights.
        saveNeuralNetwork();
    }

    /**
     * The acceptance rate
     *
     * @param currentFitness The fitness value of the current individual
     * @param newFitness     The fitness value of another individual
     * @param temperature    The temperature in the SimulatedAnnealing algorithm.
     * @return 1 if accepted.
     */
    private double acceptance(double currentFitness, double newFitness, double temperature) {
        return (newFitness < currentFitness) ? 1.0d : Math.exp((currentFitness - newFitness) / temperature);
    }
}
