package coursework;

import coursework.utility.Helpers;
import model.Fitness;
import model.Individual;
import model.NeuralNetwork;
import org.jetbrains.annotations.NotNull;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Implement an Evolutionary Algorithm (extending {@link NeuralNetwork} to solve the Lunar Landers problem.
 *
 * @author Stewart A
 */
public class ExampleEvolutionaryAlgorithm extends NeuralNetwork {

    private static double acceptance(double energy, double newEnergy, double temperature) {
        // If the new solution is better, accept it
        if (newEnergy < energy) {
            return 1.0;
        }
        // If the new solution is worse, calculate an acceptance probability
        return Math.exp((energy - newEnergy) / temperature);
    }

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

        // Initialise the population
        switch (Parameters.INITIALISATION) {
            case AUGMENTED -> population = augmented();
            case POSITIVE_NEGATIVE -> population = positiveNegative();
            case RANDOM, default -> population = initialise();
        }

        best = getBestIndividual();

        // Set the initial temperature and the cooling rate.
        double temperature = Parameters.TEMPERATURE/10;
        double cooling = 0.003;

        // The main evolutionary loop.
        while (evaluations < Parameters.maxEvaluations) {

            // Select 2 individuals from the current population
            Individual parent1;
            Individual parent2;

            switch (Parameters.SELECTION) {
                case TOURNAMENT, default -> {
                    parent1 = tournament();
                    parent2 = tournament();
                }
                case ROULETTE -> {
                    parent1 = roulette();
                    parent2 = roulette();
                }
                case ROUTE_RANK -> {
                    parent1 = rankRoutes();
                    parent2 = rankRoutes();
                }
                case RANDOM -> {
                    parent1 = random();
                    parent2 = random();
                }
            }


            // Generate children by crossover
            ArrayList<Individual> children = switch (Parameters.CROSSOVER) {
                case ARITHMETIC -> arithmetic(parent1, parent2);
                case ONE_POINT -> onePoint(parent1, parent2);
                case TWO_POINT -> twoPoint(parent1, parent2);
                case UNIFORM -> uniform(parent1, parent2);
            };

            //mutate the offspring
            switch (Parameters.MUTATION) {
                case ANNEALING -> {
                    annealing(children, temperature);
                    temperature *= 1 - cooling;
                }
                case CONSTRAINED -> constrained(children);
                case STANDARD, default -> mutate(children);
            }

            // Evaluate the children
            evaluate(children);

            // Replace children in population
            switch (Parameters.REPLACEMENT) {
                case TOURNAMENT, default -> tournament(children);
                case WORST -> worst(children);
            }
            best = getBestIndividual();
            outputStats();
        }

        //saveNeuralNetwork();
    }

    /**
     * Evaluate the fitness of the population one by one.
     * @param individuals The population
     */
    private void evaluate(@NotNull ArrayList<Individual> individuals) {
        for (Individual individual : individuals)
            individual.fitness = Fitness.evaluate(individual, this);
    }

    /**
     * Obtain the best (the lowest fitness) {@link Individual} from the current population.
     * @return The best individual
     */
    private Individual getBestIndividual() {
        best = null;
        population.stream().filter(individual -> best == null || individual.fitness < best.fitness)
                .forEachOrdered(individual -> best = individual.copy());
        return best;
    }

    /* ******************************* */
    /* ******** INITIALISATION ******* */
    /* ******************************* */

    /**
     * Initialise the population with a pseudo-random set of {@link Individual}s.
     * @return An array of {@link Individual}s.
     */
    private ArrayList<Individual> initialise() {
        population = new ArrayList<>();
        IntStream.range(0, Parameters.populationSize)
                .mapToObj(i -> new Individual()).forEachOrdered(individual -> population.add(individual));
        evaluate(population);
        return population;
    }

    /**
     * Create a larger population than the original population size and condense it down
     * to a more appropriate population size.
     *
     * @return An array of {@link Individual}s.
     */
    private ArrayList<Individual> augmented() {
        population = new ArrayList<>();
        IntStream.range(0, Parameters.populationSize + 2500)
                .mapToObj(i -> new Individual()).forEachOrdered(individual -> population.add(individual));
        evaluate(population);

        return population.stream().sorted(Comparator.naturalOrder())
                .limit(Parameters.populationSize).collect(Collectors.toCollection(ArrayList::new));
    }

    /**
     * Use positive-negative initialisation to generate a population
     * @return An array of {@link Individual}s.
     */
    private ArrayList<Individual> positiveNegative() {
        population = new ArrayList<>();

        IntStream.range(0, Parameters.populationSize).mapToObj(i -> new Individual()).forEach(individual -> {
            Individual individual2 = individual.copy();
            Arrays.setAll(individual2.chromosome, j -> 0 - individual2.chromosome[j]);
            individual.fitness = Fitness.evaluate(individual, this);
            individual2.fitness = Fitness.evaluate(individual2, this);
            population.add(individual.fitness < individual2.fitness ? individual : individual2);
        });

        return population;
    }

    /* ******************************* */
    /* ********** SELECTION ********** */
    /* ******************************* */

    /**
     * Use random selection to get an {@link Individual} from the population.
     * @return An {@link Individual}.
     */
    private Individual random() {
        Individual parent = population.get(Parameters.random.nextInt(Parameters.populationSize));
        return parent.copy();
    }

    /**
     * Use tournament selection to get an {@link Individual} from the population.
     * This function also makes use of Elitism, by copying the (or some of the)
     * best chromosomes from the current population to the new population.
     * (1). Select k solutions at random from the population.
     * (2). Select the best of these k solutions to be parents.
     * @return An {@link Individual} from the population.
     */
    private Individual tournament() {
        final int tournament = Parameters.TOURNAMENT_SIZE;
        Collections.shuffle(population);
        return population.stream().limit(tournament).min(Comparator.naturalOrder()).orElse(null);
    }

    /**
     * Use fitness-proportionate selection to obtain an {@link Individual} from the population.
     * @return An {@link Individual} from the population.
     */
    private Individual roulette() {
        // calculate the total weight
        double weightSum = 0.0;
        for (Individual c : population) {
            double v = 1 - c.fitness;
            weightSum += v;
        }

        // Generate a random number between 0 and weightSum
        double rand = weightSum * Parameters.random.nextDouble();
        // Find random value based on weights
        for (Individual indiv : population) {
            rand -= (1 - indiv.fitness);
            if (rand < 0)
                return indiv;
        }
        // When rounding errors occur, return the last item
        return population.get(-1);
    }

    /**
     * Rank the routes and select the individual that meets the selection criteria.
     * @return An {@link Individual} from the population.
     */
    private Individual rankRoutes() {

        double[] fitness = new double[Parameters.populationSize];
        for (int i = 0; i < Parameters.populationSize; i++) {
            fitness[i] = i + 1;
        }

        Helpers.unitize1(fitness);

        return population.get(Helpers.random(fitness));
    }


    /* ******************************* */
    /* ********** CROSSOVER ********** */
    /* ******************************* */

    /**
     * Use uniform crossover to create children from the population.
     * @param parent1 An {@link Individual} from the population.
     * @param parent2 An {@link Individual} from the population.
     * @return A collection of children ({@link Individual}s) from the population.
     */
    private @NotNull ArrayList<Individual> uniform(@NotNull Individual parent1, @NotNull Individual parent2) {
        Individual child1 = new Individual();
        Individual child2 = new Individual();

        int i = 0;
        if (i < parent1.chromosome.length) {
            do {
                if (Parameters.random.nextBoolean()) {
                    child1.chromosome[i] = parent1.chromosome[i];
                    child2.chromosome[i] = parent2.chromosome[i];
                } else {
                    child1.chromosome[i] = parent2.chromosome[i];
                    child2.chromosome[i] = parent1.chromosome[i];
                }
                i++;
            } while (i < parent1.chromosome.length);
        }

        return new ArrayList<>(List.of(child1, child2));
    }

    /**
     * Use One-Point crossover to create a new collection of children from the population.
     * @param parent1 An {@link Individual} from the population.
     * @param parent2 An {@link Individual} from the population.
     * @return A collection of children ({@link Individual}s) from the population.
     */
    private @NotNull ArrayList<Individual> onePoint(@NotNull Individual parent1, @NotNull Individual parent2) {
        Individual child1 = new Individual();
        Individual child2 = new Individual();
        int cut = Parameters.random.nextInt(parent1.chromosome.length);

        IntStream.range(0, parent1.chromosome.length).forEach(i -> {
            if (i < cut) {
                child1.chromosome[i] = parent1.chromosome[i];
                child2.chromosome[i] = parent2.chromosome[i];
            } else {
                child1.chromosome[i] = parent2.chromosome[i];
                child2.chromosome[i] = parent1.chromosome[i];
            }
        });

        return new ArrayList<>(List.of(child1, child2));
    }

    /**
     * Use Two-Point crossover to create a new collection of children from the population.
     * @param parent1 An {@link Individual} from the population.
     * @param parent2 An {@link Individual} from the population.
     * @return A collection of children ({@link Individual}s) from the population.
     */
    private @NotNull ArrayList<Individual> twoPoint(@NotNull Individual parent1, Individual parent2) {
        Individual child1 = new Individual();
        Individual child2 = new Individual();

        int chromLen = parent1.chromosome.length;
        int cutPoint1 = Parameters.random.nextInt(chromLen);
        int cutPoint2 = Parameters.random.nextInt((chromLen - cutPoint1) + 1) + cutPoint1;

        IntStream.range(0, chromLen).forEach(i -> {
            if (i < cutPoint1 || i >= cutPoint2) {
                child1.chromosome[i] = parent1.chromosome[i];
                child2.chromosome[i] = parent2.chromosome[i];
            } else {
                child1.chromosome[i] = parent2.chromosome[i];
                child2.chromosome[i] = parent1.chromosome[i];
            }
        });

        return new ArrayList<>(List.of(child1, child2));
    }

    /**
     * Use arithmetic crossover to create a new collection of children from the population.
     * @param parent1 An {@link Individual} from the population.
     * @param parent2 An {@link Individual} from the population.
     * @return A collection of children ({@link Individual}s) from the population.
     */
    private @NotNull ArrayList<Individual> arithmetic(@NotNull Individual parent1, Individual parent2) {
        Individual child = new Individual();
        IntStream.range(0, parent1.chromosome.length).forEach(i -> {
            double average = (parent1.chromosome[i] + parent2.chromosome[i]) / 2;
            child.chromosome[i] = average;
        });
        return new ArrayList<>(Collections.singletonList(child));
    }

    /* ******************************* */
    /* *********** MUTATION ********** */
    /* ******************************* */

    /**
     * Use standard mutation to mutate a chromosome based on a predetermined mutation rate.
     * @param individuals The population of {@link Individual}s.
     */
    private void mutate(@NotNull ArrayList<Individual> individuals) {
        for (Individual individual : individuals) {
            IntStream.range(0, individual.chromosome.length).filter(i -> Parameters.random.nextDouble() < Parameters.mutateRate).forEachOrdered(i -> {
                if (Parameters.random.nextBoolean()) {
                    individual.chromosome[i] += (Parameters.mutateChange);
                } else {
                    individual.chromosome[i] -= (Parameters.mutateChange);
                }
            });
        }
    }

    /**
     * Use constrained mutation to mutate a chromosome within a poplulation.
     * @param individuals The population of {@link Individual}s.
     */
    private void constrained(@NotNull ArrayList<Individual> individuals) {
        for (Individual individual : individuals) {
            IntStream.range(0, individual.chromosome.length).filter(i -> Parameters.random.nextDouble() < Parameters.mutateRate).forEachOrdered(i -> {
                double prior = individual.fitness;
                if (Parameters.random.nextBoolean()) {
                    individual.chromosome[i] += (Parameters.mutateChange);
                    individual.fitness = Fitness.evaluate(individual, this);
                    
                    if (individual.fitness > prior) individual.chromosome[i] -= (Parameters.mutateChange); // Undo if bad.
                } else {
                    individual.chromosome[i] -= (Parameters.mutateChange);
                    individual.fitness = Fitness.evaluate(individual, this);
                    
                    if (individual.fitness > prior) individual.chromosome[i] += (Parameters.mutateChange); // Undo if bad.
                }
            });
        }
    }

    /**
     * Taking inspiration from the {@link SimulatedAnnealing} class, this mutation operator uses
     * concepts similar to Simulated Annealing to achieve mutation.
     * @param individuals A population of {@link Individual}s.
     * @param temperature The temperature.
     */
    private void annealing(@NotNull ArrayList<Individual> individuals, double temperature) {
        for (Individual individual : individuals) {
            Individual i = individual.copy();

            // Get a random genes in the chromosome (change with next int)
            int pos1 = (int) (i.chromosome.length * Parameters.random.nextDouble());
            int pos2 = (int) (i.chromosome.length * Parameters.random.nextDouble());

            // Get the values at selected positions in the chromosome
            double swap1 = i.chromosome[pos1];
            double swap2 = i.chromosome[pos2];

            // Swap them
            i.chromosome[pos1] = swap2;
            i.chromosome[pos2] = swap1;

            // Evaluate fitness
            i.fitness = Fitness.evaluate(i, this);

            // Get energy of solutions
            double current = individual.fitness;
            double neighbour = i.fitness;

            // Decide if we should accept the neighbour
            if (acceptance(current, neighbour, temperature)
                    >= Parameters.random.nextDouble()) individual = i;
        }
    }


    /**
     * Replace the worst {@link Individual}s in the population with new random {@link Individual}s
     * @param individuals A collection of {@link Individual}s from the population.
     */
    private void worst(@NotNull ArrayList<Individual> individuals) {
        individuals.forEach(individual -> {
            int index = getWorst();
            population.set(index, individual);
        });
    }

    /**
     * Similar to selection, but replace the worst {@link Individual}s in the population.
     * @param individuals A collection of {@link Individual}s
     */
    private void tournament(@NotNull ArrayList<Individual> individuals) {
        final int tournament = Parameters.TOURNAMENT_SIZE;

        individuals.forEach(individual -> {
            Collections.shuffle(population);
            Individual worst = population.stream().limit(tournament).max(Comparator.naturalOrder()).orElse(null);
            population.remove(worst);
            population.add(individual);
        });
    }

    /**
     * Obtain the index of the worst performing {@link Individual} in the population. 
     * @return The index of the worst performing {@link Individual}s
     */
    private int getWorst() {
        Individual worst = null;
        int index = -1;
        for (int i = 0; i < population.size(); i++) {
            Individual individual = population.get(i);
            if (worst == null || individual.fitness > worst.fitness) {
                worst = individual;
                index = i;
            }
        }
        return index;
    }
}
