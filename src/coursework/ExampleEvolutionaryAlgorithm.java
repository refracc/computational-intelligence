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
        return Helpers.activationFunction(v);
    }

    @Override
    public void run() {
        switch(Parameters.INITIALISATION) {
            case AUGMENTED -> population = augmented();
            case POSITIVE_NEGATIVE -> population = positiveNegative();
            case RANDOM -> population = initialise();
        }

        best = getBestIndividual();

        double temperature = Parameters.TEMPERATURE;
        double cooling = Parameters.COOLING_RATE;

        while (Parameters.maxEvaluations > evaluations) {
            Individual i1 = new Individual();
            Individual i2 = new Individual();

            switch (Parameters.SELECTION) {
                case RANDOM -> {
                    i1 = random();
                    i2 = random();
                }
                case ROULETTE -> {
                    i1 = roulette();
                    i2 = roulette();
                }
                case ROUTE_RANK -> {
                    i1 = ranked();
                    i2 = ranked();
                }
                case TOURNAMENT -> {
                    i1 = tournament();
                    i2 = tournament();
                }
            }

            ArrayList<Individual> children = new ArrayList<>();
            switch (Parameters.CROSSOVER) {
                case ARITHMETIC -> children = arithmetic(i1, i2);
                case ONE_POINT -> children = onePoint(i1, i2);
                case TWO_POINT -> children = twoPoint(i1, i2);
                case UNIFORM -> children = uniform(i1, i2);
            }

            switch(Parameters.MUTATION) {
                case ANNEALING -> {
                    annealing(children, temperature);
                    temperature *= (1 - cooling);
                }
                case CONSTRAINED -> constrained(children);
                case STANDARD -> mutate(children);
            }

            evaluatePopulation(children);

            switch (Parameters.REPLACEMENT) {
                case TOURNAMENT -> tournament(children);
                case WORST -> worst(children);
            }

            best = getBestIndividual();
            outputStats();
        }
        saveNeuralNetwork();
    }

    /**
     * Evaluate the fitness of the population.
     *
     * @param individuals The list of individuals (population) to evaluate.
     */
    private void evaluatePopulation(@NotNull ArrayList<Individual> individuals) {
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
        best = random();
        for (Individual i : population) {
            if (best.fitness > i.fitness) {
                best = i;
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
    private ArrayList<Individual> initialise() {
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
    private ArrayList<Individual> augmented() {
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
    private ArrayList<Individual> positiveNegative() {
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
     *
     * @param individual1 The first parent.
     * @param individual2 The second parent.
     * @return The new children post-crossover.
     */
    private @NotNull ArrayList<Individual> uniform(@NotNull Individual individual1, @NotNull Individual individual2) {
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

        return new ArrayList<>(Arrays.asList(i1, i2));
    }

    /**
     * Perform one-point crossover between 2 individuals.
     *
     * @param individual1 The first parent.
     * @param individual2 The second parent.
     * @return The new children post-crossover.
     */
    private @NotNull ArrayList<Individual> onePoint(@NotNull Individual individual1, @NotNull Individual individual2) {
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

        return new ArrayList<>(Arrays.asList(i1, i2));
    }

    /**
     * Perform two-point crossover between 2 individuals.
     *
     * @param individual1 The first parent.
     * @param individual2 The second parent.
     * @return The new children post-crossover.
     */
    private @NotNull ArrayList<Individual> twoPoint(@NotNull Individual individual1, @NotNull Individual individual2) {
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

        return new ArrayList<>(Arrays.asList(i1, i2));
    }

    /**
     * Perform arithmetic crossover between 2 individuals.
     *
     * @param individual1 The first parent.
     * @param individual2 The second parent.
     * @return The new children post-crossover.
     */
    private @NotNull @Unmodifiable ArrayList<Individual> arithmetic(@NotNull Individual individual1, @NotNull Individual individual2) {
        Individual individual = new Individual();

        IntStream.range(0, individual1.chromosome.length).forEach(i -> {
            double average = (individual1.chromosome[i] + individual2.chromosome[i]) / 2;
            individual.chromosome[i] = average;
        });

        return new ArrayList<>(Collections.singletonList(individual));
    }

    /* ********************************** */
    /* *********** MUTATION ************* */
    /* ********************************** */

    /**
     * Perform mutation among a population of individuals.
     *
     * @param population The population to have mutation applied to it.
     */
    private void mutate(@NotNull ArrayList<Individual> population) {
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
     *
     * @param population The population to have constrained mutation applied to it.
     */
    private void constrained(@NotNull ArrayList<Individual> population) {
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
     *
     * @param population  The population to mutate.
     * @param temperature The temperature.
     */
    @Contract(pure = true)
    private void annealing(@NotNull ArrayList<Individual> population, double temperature) {
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
     *
     * @param population The population of individuals.
     */
    private void tournament(@NotNull ArrayList<Individual> population) {
        for (int i = 0; i < population.size(); i++) {
            Collections.shuffle(population);
            Individual ind = population.get(i);
            Individual worst = population.stream().limit(Parameters.TOURNAMENT_SIZE).max(Comparator.naturalOrder()).orElse(null);

            population.removeAll(List.of(Objects.requireNonNull(worst)));
            population.add(ind);
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
     *
     * @param population The population of individuals.
     */
    private void worst(@NotNull ArrayList<Individual> population) {
        for (Individual individual : population) {
            int index = getWorstIndex();
            population.set(index, individual);
        }
    }


}
