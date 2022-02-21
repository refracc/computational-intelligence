package coursework;

import coursework.utility.Helpers;
import model.Fitness;
import model.Individual;
import model.NeuralNetwork;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Implement an Evolutionary Algorithm (extending {@link NeuralNetwork} to solve the Lunar Landers problem.
 *
 * @author Stewart A
 */
public class ExampleEvolutionaryAlgorithm extends NeuralNetwork {

    private static double acceptanceProbability(double energy, double newEnergy, double temperature) {
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

        //Initialise a population of Individuals with random weights
        switch (Parameters.INITIALISATION) {
            case AUGMENTED -> population = augmentedInitialise();
            case POSITIVE_NEGATIVE -> population = PosNegInitialise();
            case RANDOM, default -> population = initialise();
        }

        best = getBest();

        // Set initial temp, cooling rate, and improvement count
        double temp = Parameters.TEMPERATURE/10;
        double coolingRate = 0.003;

        // main EA processing loop
        while (evaluations < Parameters.maxEvaluations) {

            // Select 2 Individuals from the current population
            Individual parent1;
            Individual parent2;

            switch (Parameters.SELECTION) {
                case TOURNAMENT, default -> {
                    parent1 = tournamentSelect();
                    parent2 = tournamentSelect();
                }
                case ROULETTE -> {
                    parent1 = rouletteSelect();
                    parent2 = rouletteSelect();
                }
                case ROUTE_RANK -> {
                    parent1 = rankSelect();
                    parent2 = rankSelect();
                }
                case RANDOM -> {
                    parent1 = randSelect();
                    parent2 = randSelect();
                }
            }


            // Generate children by crossover
            ArrayList<Individual> children = switch (Parameters.CROSSOVER) {
                case ARITHMETIC -> arithmeticCrossover(parent1, parent2);
                case ONE_POINT -> onePointCrossover(parent1, parent2);
                case TWO_POINT -> twoPointCrossover(parent1, parent2);
                case UNIFORM -> uniformCrossover(parent1, parent2);
            };

            //mutate the offspring
            switch (Parameters.MUTATION) {
                case ANNEALING -> {
                    mutateAnnealing(children, temp);
                    temp *= 1 - coolingRate;
                }
                case CONSTRAINED -> constrainedMutation(children);
                case STANDARD, default -> mutate(children);
            }

            // Evaluate the children
            evaluateIndividuals(children);

            // Replace children in population
            switch (Parameters.REPLACEMENT) {
                case TOURNAMENT, default -> tournamentReplace(children);
                case WORST -> replaceWorst(children);
            }
            best = getBest();
            outputStats();
        }

        saveNeuralNetwork();  // save the trained network to disk
    }

    /**
     * Sets the fitness of the individuals passed as parameters (whole population)
     */
    private void evaluateIndividuals(ArrayList<Individual> individuals) {
        for (Individual individual : individuals) {
            individual.fitness = Fitness.evaluate(individual, this);
        }
    }

    /**
     * Returns a copy of the best individual in the population
     */
    private Individual getBest() {
        best = null;
        for (Individual individual : population) {
            if (best == null) {
                best = individual.copy();
            } else if (individual.fitness < best.fitness) {
                best = individual.copy();
            }
        }
        return best;
    }

    /**
     * INITIALISATION. Generates a randomly initialised population
     */
    private ArrayList<Individual> initialise() {
        population = new ArrayList<>();
        for (int i = 0; i < Parameters.populationSize; ++i) {
            // chromosome weights are initialised randomly in the constructor
            Individual individual = new Individual();
            population.add(individual);
        }
        evaluateIndividuals(population);
        return population;
    }

    private ArrayList<Individual> augmentedInitialise() {
        population = new ArrayList<>();
        for (int i = 0; i < Parameters.populationSize + 1000; i++) {
            // chromosome weights are initialised randomly in the constructor
            Individual individual = new Individual();
            population.add(individual);
        }
        evaluateIndividuals(population);

        return population
                .stream()
                .sorted(Comparator.naturalOrder())
                .limit(Parameters.populationSize)
                .collect(Collectors.toCollection(ArrayList::new));
    }

    private ArrayList<Individual> PosNegInitialise() {
        population = new ArrayList<>();
        for (int i = 0; i < Parameters.populationSize; ++i) {
            // chromosome weights are initialised randomly in the constructor
            Individual individual = new Individual();
            Individual individual2 = individual.copy();

            for (int j = 0; j < individual2.chromosome.length; j++) {
                // Flip chromes
                individual2.chromosome[j] = 0 - individual2.chromosome[j];
            }
            individual.fitness = Fitness.evaluate(individual, this);
            individual2.fitness = Fitness.evaluate(individual2, this);

            if (individual.fitness < individual2.fitness) {
                population.add(individual);
            } else {
                population.add(individual2);
            }
        }
        return population;
    }

    /**
     * SELECTION
     */
    private Individual randSelect() {
        Individual parent = population.get(Parameters.random.nextInt(Parameters.populationSize));
        return parent.copy();
    }

    private Individual tournamentSelect() {
        /*
          Elitism - copy the best chromosome (or a few best chromosomes) to new population
          (happens if tournament size is equal to total pop size)
          1 - Pick t solutions completely at random from the population
          2 - Select the best of the t solutions to be a parent
         */
        final int TOURNAMET_SIZE = Parameters.TOURNAMENT_SIZE;

        Collections.shuffle(population);
        Individual parent = population
                .stream()
                .limit(TOURNAMET_SIZE).min(Comparator.naturalOrder())
                .orElse(null);
        return parent;
    }

    // Fitness proportionate selection - roulette wheel selection
    private Individual rouletteSelect() {
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

    private Individual rankSelect() {

        double[] fitness = new double[Parameters.populationSize];
        for (int i = 0; i < Parameters.populationSize; i++) {
            fitness[i] = i + 1;
        }

        Helpers.unitize1(fitness);

        return population.get(Helpers.random(fitness));
    }

    /**
     * CROSSOVER
     */
    private @NotNull ArrayList<Individual> uniformCrossover(@NotNull Individual parent1, @NotNull Individual parent2) {
        Individual child1 = new Individual();
        Individual child2 = new Individual();

        int i = 0;
        while (i < parent1.chromosome.length) {
            if (Parameters.random.nextBoolean()) {
                child1.chromosome[i] = parent1.chromosome[i];
                child2.chromosome[i] = parent2.chromosome[i];
            } else {
                child1.chromosome[i] = parent2.chromosome[i];
                child2.chromosome[i] = parent1.chromosome[i];
            }
            i++;
        }

        ArrayList<Individual> children = new ArrayList<>();
        children.add(child1);
        children.add(child2);
        return children;
    }

    private @NotNull ArrayList<Individual> onePointCrossover(@NotNull Individual parent1, @NotNull Individual parent2) {
        Individual child1 = new Individual();
        Individual child2 = new Individual();
        int cutPoint = Parameters.random.nextInt(parent1.chromosome.length);

        for (int i = 0; i < parent1.chromosome.length; i++) {
            if (i < cutPoint) {
                child1.chromosome[i] = parent1.chromosome[i];
                child2.chromosome[i] = parent2.chromosome[i];
            } else {
                child1.chromosome[i] = parent2.chromosome[i];
                child2.chromosome[i] = parent1.chromosome[i];
            }
        }

        ArrayList<Individual> children = new ArrayList<>();
        children.add(child1);
        children.add(child2);
        return children;
    }

    private @NotNull ArrayList<Individual> twoPointCrossover(@NotNull Individual parent1, Individual parent2) {
        Individual child1 = new Individual();
        Individual child2 = new Individual();

        int chromLen = parent1.chromosome.length;
        int cutPoint1 = Parameters.random.nextInt(chromLen);
        int cutPoint2 = Parameters.random.nextInt((chromLen - cutPoint1) + 1) + cutPoint1;

        for (int i = 0; i < chromLen; i++) {
            if (i < cutPoint1 || i >= cutPoint2) {
                child1.chromosome[i] = parent1.chromosome[i];
                child2.chromosome[i] = parent2.chromosome[i];
            } else {
                child1.chromosome[i] = parent2.chromosome[i];
                child2.chromosome[i] = parent1.chromosome[i];
            }
        }

        ArrayList<Individual> children = new ArrayList<>();
        children.add(child1);
        children.add(child2);
        return children;
    }

    private @NotNull ArrayList<Individual> arithmeticCrossover(@NotNull Individual parent1, Individual parent2) {
        Individual child = new Individual();
        for (int i = 0; i < parent1.chromosome.length; i++) {
            double avgChrom = (parent1.chromosome[i] + parent2.chromosome[i]) / 2;
            child.chromosome[i] = avgChrom;
        }
        ArrayList<Individual> children = new ArrayList<>();
        children.add(child);
        return children;
    }

    /**
     * MUTATION
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

    private void constrainedMutation(@NotNull ArrayList<Individual> individuals) {
        for (Individual individual : individuals) {
            IntStream.range(0, individual.chromosome.length).filter(i -> Parameters.random.nextDouble() < Parameters.mutateRate).forEachOrdered(i -> {
                if (Parameters.random.nextBoolean()) {
                    double oldFitness = individual.fitness;
                    individual.chromosome[i] += (Parameters.mutateChange);
                    individual.fitness = Fitness.evaluate(individual, this);
                    if (individual.fitness > oldFitness) {
                        // revert if bad choice was made
                        individual.chromosome[i] -= (Parameters.mutateChange);
                    }
                } else {
                    double oldFitness = individual.fitness;
                    individual.chromosome[i] -= (Parameters.mutateChange);
                    individual.fitness = Fitness.evaluate(individual, this);
                    if (individual.fitness > oldFitness) {
                        // revert if bad choice was made
                        individual.chromosome[i] += (Parameters.mutateChange);
                    }
                }
            });
        }
    }

    private void mutateAnnealing(@NotNull ArrayList<Individual> individuals, double temp) {
        for (Individual individual : individuals) {
            Individual newIndividual = individual.copy();

            // Get a random genes in the chromosome (change with next int)
            int chromeGenePos1 = (int) (newIndividual.chromosome.length * Parameters.random.nextDouble());
            int chromeGenePos2 = (int) (newIndividual.chromosome.length * Parameters.random.nextDouble());

            // Get the values at selected positions in the chromosome
            double geneSwap1 = newIndividual.chromosome[chromeGenePos1];
            double geneSwap2 = newIndividual.chromosome[chromeGenePos2];

            // Swap them
            newIndividual.chromosome[chromeGenePos1] = geneSwap2;
            newIndividual.chromosome[chromeGenePos2] = geneSwap1;

            // Evaluate fitness
            newIndividual.fitness = Fitness.evaluate(newIndividual, this);

            // Get energy of solutions
            double currentEnergy = individual.fitness;
            double neighbourEnergy = newIndividual.fitness;

            // Decide if we should accept the neighbour
            if (acceptanceProbability(currentEnergy, neighbourEnergy, temp)
                    >= Parameters.random.nextDouble()) {
                individual = newIndividual;
            }
        }
    }


    /**
     * REPLACEMENT
     */
    private void replaceWorst(@NotNull ArrayList<Individual> individuals) {
        for (Individual individual : individuals) {
            int idx = getWorstIndex();
            population.set(idx, individual);
        }
    }

    // Replace using tournament - same as selection but with worst
    private void tournamentReplace(@NotNull ArrayList<Individual> individuals) {
        final int TOURNAMET_SIZE = Parameters.TOURNAMENT_SIZE;

        for (Individual individual : individuals) {
            Collections.shuffle(population);
            Individual worstChrom = population
                    .stream()
                    .limit(TOURNAMET_SIZE).max(Comparator.naturalOrder())
                    .orElse(null);

            population.remove(worstChrom);
            population.add(individual);
        }
    }

    // Returns the index of the worst member of the population
    private int getWorstIndex() {
        Individual worst = null;
        int idx = -1;
        for (int i = 0; i < population.size(); i++) {
            Individual individual = population.get(i);
            if (worst == null) {
                worst = individual;
                idx = i;
            } else if (individual.fitness > worst.fitness) {
                worst = individual;
                idx = i;
            }
        }
        return idx;
    }
}
