package coursework;

import model.Fitness;
import model.Individual;
import model.NeuralNetwork;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Implement an Evolutionary Algorithm (extending {@link NeuralNetwork} to solve the Lunar Landers problem.
 *
 * @author Stewart A
 */
public class ExampleEvolutionaryAlgorithm extends NeuralNetwork {

    /**
     * The function that will be setting up the Activation Function for the Neural Network
     *
     * @param v A {@link Double} value that is given by the {@link NeuralNetwork}.
     * @return Some value based on the Activation Type set out in the {@link Parameters} class.
     */
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

    }

    /**
     * Evaluate the fitness of the population.
     *
     * @param individuals The list of individuals (population) to evaluate.
     */
    private void evaluatePopulation(List<Individual> individuals) {
        for (Individual i : individuals) {
            i.fitness = Fitness.evaluate(i, this);
        }
    }

    /**
     * Find the best (the lowest fitness) individual of the population.
     *
     * @return The best performing individual.
     */
    private Individual getBestIndividual() {
        best = null;
        for (Individual i : population) {
            if (best.fitness > i.fitness) {
                best = i.copy();
            }
        }

        return best;
    }

    /**
     * Generate a randomly-initialised population for the Evolutionary Algorithm.
     *
     * @return A randomly-initialised population.
     */
    private List<Individual> initialise() {
        population = new ArrayList<>();

        // Create new population
        // Has weights initialised randomly within the constructor.
        for (int i = 0; i < Parameters.populationSize; i++) {
            Individual individual = new Individual();
            population.add(individual);
        }

        evaluatePopulation(population);
        return population;
    }


    /**
     * Generate a larger randomly-initialised population for the Evolutionary Algorithm.
     *
     * @return A randomly-initialised population.
     */
    private List<Individual> initialiseAugmented() {
        population = new ArrayList<>();

        IntStream.range(0, Parameters.populationSize + 5000).mapToObj(i -> new Individual()).forEach(individual -> population.add(individual));

        // Evaluate population
        evaluatePopulation(population);
        // Reduce the population down to the proper population size after comparison & evaluation.
        return population.stream().sorted(Individual::compareTo).limit(Parameters.populationSize).collect(Collectors.toCollection(ArrayList::new));
    }

    /**
     * Generate a population using positive/negative for the Evolutionary Algorithm.
     *
     * @return A randomly-initialised population.
     */
    private List<Individual> initialisePositiveNegative() {
        population = new ArrayList<>();

        for (int i = 0; i < Parameters.populationSize; i++) {
            Individual i1 = new Individual();
            Individual i2 = i1.copy();

            // Flip
            Arrays.setAll(i2.chromosome, j -> (0 - i2.chromosome[j]));

            i1.fitness = Fitness.evaluate(i1, this);
            i2.fitness = Fitness.evaluate(i2, this);

            population.add(i2.fitness > i1.fitness ? i1 : i2);
        }
        return population;
    }

    /**
     * Generate part of a randomly-initialised population for the Evolutionary Algorithm.
     */
    private void initialisePartially() {
        Individual individual;
        population.sort(Comparator.reverseOrder());

        for (int i = 0; i < 15; i++) {
            individual = new Individual();
            population.remove(0);
            population.add(individual);
        }
        evaluatePopulation(population);
    }

    /**
     * Keep the best N individuals of a population for evaluation.
     *
     * @param n The number of individuals to keep.
     */
    private void keepBestNIndividuals(int n) {
        population.sort(Comparator.reverseOrder());

        IntStream.range(0, Parameters.populationSize - n).mapToObj(i -> new Individual()).forEach(individual -> {
            population.remove(0);
            population.add(individual);
        });

        evaluatePopulation(population);
    }
}
