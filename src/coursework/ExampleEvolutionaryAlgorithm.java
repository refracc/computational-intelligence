package coursework;

import coursework.utility.Helpers;
import model.Fitness;
import model.Individual;
import model.NeuralNetwork;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Unmodifiable;

import java.util.*;
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
    private void evaluatePopulation(@NotNull List<Individual> individuals) {
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

    /* ********************************** */
    /* ********* INITIALISATION ********* */
    /* ********************************** */

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
    private List<Individual> augmented() {
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
    private List<Individual> positiveNegative() {
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
    private void partially() {
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

    /* ********************************** */
    /* *********** SELECTION ************ */
    /* ********************************** */

    /**
     * Select a random individual from the population.
     *
     * @return A random individual from the population.
     */
    private Individual random() {
        Individual individual = population.get(Parameters.random.nextInt(Parameters.populationSize));
        return individual.copy();
    }

    /**
     * Perform tournament selection on the population.
     *
     * @return An individual from the population.
     */
    private Individual tournament() {
        Collections.shuffle(population);
        return population.stream().limit(Parameters.TOURNAMENT_SIZE).max(Comparator.naturalOrder()).orElse(null);
    }

    /**
     * Perform roulette selection on the population.
     * Also known as Fitness-Proportionate Selection.
     *
     * @return An individual.
     */
    private Individual roulette() {
        double sum = population.stream().mapToDouble(i -> 1 - i.fitness).sum();
        double r = sum * Parameters.random.nextDouble();

        for (Individual i : population) {
            r -= (1 - i.fitness);
            if (r < 0) return i;
        }
        // Should a rounding error occur, the last element in the population will be returned.
        return population.get(-1);
    }

    /**
     * Get an individual based on Ranked Fitness Proportionate.
     * @return The last individual in the list (if the for)
     */
    private Individual oldRanked() {
        population.sort(Comparator.reverseOrder());

        int random = Parameters.random.nextInt(Parameters.populationSize);
        for (int i = 0; i < Parameters.populationSize; i++) {
            random--;
            if (i > random) return population.get(i);
        }

        return population.get(-1);
    }

    /**
     * Get an individual by using Ranked Selection
     * @return An individual
     */
    private Individual ranked() {
        double[] fitness = IntStream.range(0, Parameters.populationSize).mapToDouble(i -> i + 1).toArray();

        Helpers.unitize1(fitness);
        return population.get(Helpers.random(fitness));
    }

    /* ********************************** */
    /* *********** CROSSOVER ************ */
    /* ********************************** */

    /**
     * Perform uniform crossover between 2 individuals.
     * @param individual1 The first parent.
     * @param individual2 The second parent.
     * @return The new children post-crossover.
     */
    private @NotNull List<Individual> uniform(@NotNull Individual individual1, @NotNull Individual individual2) {
        Individual i1 = new Individual();
        Individual i2 = new Individual();

        IntStream.range(0, individual1.chromosome.length).forEach(i -> {
            if (Parameters.random.nextBoolean()) {
                i1.chromosome[i] = individual1.chromosome[i];
                i2.chromosome[i] = individual2.chromosome[i];
            } else {
                i1.chromosome[i] = individual2.chromosome[i];
                i2.chromosome[i] = individual1.chromosome[i];
            }
        });

        return Arrays.asList(i1, i2);
    }

    /**
     * Perform one-point crossover between 2 individuals.
     * @param individual1 The first parent.
     * @param individual2 The second parent.
     * @return The new children post-crossover.
     */
    private @NotNull List<Individual> onePoint(@NotNull Individual individual1, @NotNull Individual individual2) {
        Individual i1 = new Individual();
        Individual i2 = new Individual();

        int cut = Parameters.random.nextInt(i1.chromosome.length);

        IntStream.range(0, individual1.chromosome.length).forEach(i -> {
            if (i < cut) {
                i1.chromosome[i] = individual1.chromosome[i];
                i2.chromosome[i] = individual2.chromosome[i];
            } else {
                i1.chromosome[i] = individual2.chromosome[i];
                i2.chromosome[i] = individual1.chromosome[i];
            }
        });

        return Arrays.asList(i1, i2);
    }

    /**
     * Perform two-point crossover between 2 individuals.
     * @param individual1 The first parent.
     * @param individual2 The second parent.
     * @return The new children post-crossover.
     */
    private @NotNull List<Individual> twoPoint(@NotNull Individual individual1, @NotNull Individual individual2) {
        Individual i1 = new Individual();
        Individual i2 = new Individual();
        int cut1 = Parameters.random.nextInt(i1.chromosome.length);
        int cut2 = Parameters.random.nextInt(i1.chromosome.length);

        IntStream.range(0, individual1.chromosome.length).forEach(i -> {
            if ((i < cut1) || (i >= cut2)) {
                i1.chromosome[i] = individual1.chromosome[i];
                i2.chromosome[i] = individual2.chromosome[i];
            } else {
                i1.chromosome[i] = individual2.chromosome[i];
                i2.chromosome[i] = individual1.chromosome[i];
            }
        });

        return Arrays.asList(i1, i2);
    }

    /**
     * Perform arithmetic crossover between 2 individuals.
     * @param individual1 The first parent.
     * @param individual2 The second parent.
     * @return The new children post-crossover.
     */
    private @NotNull @Unmodifiable List<Individual> arithmetic(@NotNull Individual individual1, @NotNull Individual individual2) {
        Individual individual = new Individual();

        IntStream.range(0, individual1.chromosome.length).forEach(i -> {
           double average = (individual1.chromosome[i] + individual2.chromosome[i]) / 2;
           individual.chromosome[i] = average;
        });

        return Collections.singletonList(individual);
    }

    /* ********************************** */
    /* *********** MUTATION ************* */
    /* ********************************** */

    /**
     * Perform mutation among a population of individuals.
     * @param population The population to have mutation applied to it.
     */
    private void mutate(@NotNull List<Individual> population) {
        population.forEach(individual -> IntStream.range(0, individual.chromosome.length).forEach(i -> {
            if (Parameters.random.nextDouble() < Parameters.mutateRate) {
                individual.chromosome[i] = (Parameters.random.nextBoolean())
                        ? individual.chromosome[i] + Parameters.mutateRate
                        : individual.chromosome[i] - Parameters.mutateRate;
            }
        }));
    }

    /**
     * Perform constrained mutation among a population of individuals.
     * @param population The population to have constrained mutation applied to it.
     */
    private void constrained(@NotNull List<Individual> population) {
        population.forEach(individual -> IntStream.range(0, individual.chromosome.length).forEach(i -> {
            if (Parameters.random.nextDouble() < Parameters.mutateRate) {
                if (Parameters.random.nextBoolean()) {
                    double prior = individual.fitness;
                    individual.chromosome[i] += Parameters.mutateChange;
                    individual.fitness = Fitness.evaluate(individual, this);

                    individual.chromosome[i] = (individual.fitness > prior)
                            ? individual.chromosome[i] - Parameters.mutateChange
                            : individual.chromosome[i]; // Revert if a bad choice was made.
                } else {
                    double prior = individual.fitness;
                    individual.chromosome[i] -= (Parameters.mutateChange);
                    individual.fitness = Fitness.evaluate(individual, this);

                    individual.chromosome[i] = (individual.fitness > prior)
                            ? individual.chromosome[i] + Parameters.mutateChange
                            : individual.chromosome[i]; // Revert if a bad choice was made.
                }
            }
        }));
    }

    /**
     * The acceptance probability for taking a new solution.
     * @param currentFitness The current fitness value for comparison.
     * @param newFitness The new fitness value for comparison.
     * @param temperature The temperature value for comparison.
     * @return Either a 100% acceptance, or some value {@code 0 <= x <= 1}.
     */
    private double acceptance(double currentFitness, double newFitness, double temperature) {
        return (newFitness < currentFitness) ? 1.0d : Math.exp((currentFitness - newFitness) / temperature);
    }

    /**
     * Mutate the population with inspiration from SimulatedAnnealing.
     * @param population The population to mutate.
     * @param temperature The temperature.
     */
    @Contract(pure = true)
    private void annealing(@NotNull List<Individual> population, double temperature) {
        for (Individual individual : population) {
            Individual i = individual.copy();

            int pos1 = (int) (i.chromosome.length * Parameters.random.nextDouble());
            int pos2 = (int) (i.chromosome.length * Parameters.random.nextDouble());

            double swap1 = i.chromosome[pos1];
            double swap2 = i.chromosome[pos2];

            i.chromosome[pos1] = swap2;
            i.chromosome[pos2] = swap1;

            i.fitness = Fitness.evaluate(i, this);

            double fitness = individual.fitness;
            double neighbouringFitness = i.fitness;

            if (Parameters.random.nextDouble() <= acceptance(fitness, neighbouringFitness, temperature)) individual = i;
        }
    }

    /* ********************************** */
    /* ********** REPLACEMENT *********** */
    /* ********************************** */

    /**
     * Perform tournament replacement on a population.
     * @param population The population of individuals.
     */
    private void tournament(@NotNull List<Individual> population) {
        for (Individual individual : population) {
            Collections.shuffle(population);
            Individual worst = population.stream().limit(Parameters.TOURNAMENT_SIZE).max(Comparator.naturalOrder()).orElse(null);

            population.remove(worst);
            population.add(individual);
        }
    }

    /**
     * Helper function to find the index of the worst performing individual in the population.
     */
    private int getWorstIndex() {
        Individual worst = null;
        int index = -1;

        for (int i = 0; i < population.size(); i++) {
            Individual individual = population.get(i);

            if ((worst == null) || (worst.fitness < individual.fitness)) {
                worst = individual;
                index = i;
            }
        }
        return index;
    }

    /**
     * Find the worst performing individual within a population and have it replaced.
     * @param population The population of individuals.
     */
    private void worst(@NotNull List<Individual> population) {
        for (Individual individual : population) {
            int index = getWorstIndex();
            population.set(index, individual);
        }
    }


}
