package coursework

import model.NeuralNetwork
import coursework.options.Initialisation
import model.Individual
import coursework.options.Selection
import coursework.options.Crossover
import coursework.options.Mutation
import coursework.options.Replacement
import coursework.utility.Helpers
import model.Fitness
import java.util.stream.IntStream
import java.util.stream.Collectors
import java.util.Arrays
import java.util.Comparator
import java.util.function.Consumer

/**
 * Implement an Evolutionary Algorithm (extending [NeuralNetwork]) to solve the Lunar Landers problem.
 *
 * @author Stewart A
 */
class ExampleEvolutionaryAlgorithm : NeuralNetwork() {
    /**
     * The function that will be setting up the Activation Function for the Neural Network
     *
     * @param v A [Double] value that is given by the [NeuralNetwork].
     * @return Some value based on the Activation Type set out in the [Parameters] class.
     */
    override fun activationFunction(v: Double): Double {
        return Helpers.activationFunction(v)
    }

    override fun run() {

        // Initialise the population
        population = when (Parameters.INITIALISATION) {
            Initialisation.AUGMENTED -> augmented()
            Initialisation.POSITIVE_NEGATIVE -> positiveNegative()
            Initialisation.RANDOM -> initialise()
        }
        best = bestIndividual

        // Set the initial temperature and the cooling rate.
        var temperature = Parameters.TEMPERATURE / 10
        val cooling = 0.003

        // The main evolutionary loop.
        while (evaluations < Parameters.maxEvaluations) {

            // Select 2 individuals from the current population
            var parent1: Individual
            var parent2: Individual
            when (Parameters.SELECTION) {
                Selection.TOURNAMENT -> {
                    parent1 = tournament()
                    parent2 = tournament()
                }
                Selection.ROULETTE -> {
                    parent1 = roulette()
                    parent2 = roulette()
                }
                Selection.ROUTE_RANK -> {
                    parent1 = rankRoutes()
                    parent2 = rankRoutes()
                }
                Selection.RANDOM -> {
                    parent1 = random()
                    parent2 = random()
                }
            }


            // Generate children by crossover
            val children: java.util.ArrayList<Individual> = when (Parameters.CROSSOVER) {
                Crossover.ARITHMETIC -> arithmetic(parent1, parent2)
                Crossover.ONE_POINT -> onePoint(parent1, parent2)
                Crossover.TWO_POINT -> twoPoint(parent1, parent2)
                Crossover.UNIFORM -> uniform(parent1, parent2)
            }

            // Mutate the children
            when (Parameters.MUTATION) {
                Mutation.ANNEALING -> {
                    annealing(children, temperature)
                    temperature *= 1 - cooling
                }
                Mutation.CONSTRAINED -> constrained(children)
                Mutation.STANDARD -> mutate(children)
            }

            // Evaluate the children
            evaluate(children)
            when (Parameters.REPLACEMENT) {
                Replacement.TOURNAMENT -> tournament(children)
                Replacement.WORST -> worst(children)
            }
            best = bestIndividual
            outputStats()
        }
        saveNeuralNetwork()
    }

    /**
     * Evaluate the fitness of the population one by one.
     * @param individuals The population
     */
    private fun evaluate(individuals: java.util.ArrayList<Individual>) {
        for (individual in individuals) individual.fitness = Fitness.evaluate(individual, this)
    }

    /**
     * Obtain the best (the lowest fitness) [Individual] from the current population.
     * @return The best individual
     */
    private val bestIndividual: Individual
        get() {
            best = null
            for (individual in population) {
                if (best == null || individual.fitness < best.fitness) {
                    best = individual.copy()
                }
            }
            return best
        }
    /* ******************************* */ /* ******** INITIALISATION ******* */ /* ******************************* */
    /**
     * Initialise the population with a pseudo-random set of [Individual]s.
     * @return An array of [Individual]s.
     */
    private fun initialise(): java.util.ArrayList<Individual> {
        population = java.util.ArrayList()
        IntStream.range(0, Parameters.populationSize)
            .mapToObj { Individual() }
            .forEachOrdered { individual: Individual? -> population.add(individual) }
        evaluate(population)
        return population
    }

    /**
     * Create a larger population than the original population size and condense it down
     * to a more appropriate population size.
     *
     * @return An array of [Individual]s.
     */
    private fun augmented(): java.util.ArrayList<Individual> {
        population = java.util.ArrayList()
        IntStream.range(0, Parameters.populationSize + 2500)
            .mapToObj { Individual() }
            .forEachOrdered { individual: Individual? -> population.add(individual) }
        evaluate(population)
        return population.stream().sorted(Comparator.naturalOrder())
            .limit(Parameters.populationSize.toLong()).collect(
                Collectors.toCollection { ArrayList() }
            )
    }

    /**
     * Use positive-negative initialisation to generate a population
     * @return An array of [Individual]s.
     */
    private fun positiveNegative(): java.util.ArrayList<Individual> {
        population = java.util.ArrayList()
        IntStream.range(0, Parameters.populationSize).mapToObj { Individual() }
            .forEach { individual: Individual ->
                val individual2 = individual.copy()
                Arrays.setAll(individual2.chromosome) { j: Int -> 0 - individual2.chromosome[j] }
                individual.fitness = Fitness.evaluate(individual, this)
                individual2.fitness = Fitness.evaluate(individual2, this)
                if (individual.fitness < individual2.fitness) population.add(individual) else population.add(individual2)
            }
        return population
    }
    /* ******************************* */ /* ********** SELECTION ********** */ /* ******************************* */
    /**
     * Use random selection to get an [Individual] from the population.
     * @return An [Individual].
     */
    private fun random(): Individual {
        val parent = population[Parameters.random.nextInt(Parameters.populationSize)]
        return parent.copy()
    }

    /**
     * Use tournament selection to get an [Individual] from the population.
     * This function also makes use of Elitism, by copying the (or some of the)
     * best chromosomes from the current population to the new population.
     * (1). Select k solutions at random from the population.
     * (2). Select the best of these k solutions to be parents.
     * @return An [Individual] from the population.
     */
    private fun tournament(): Individual {
        val tournament = Parameters.TOURNAMENT_SIZE
        population.shuffle()
        return population.stream().limit(tournament.toLong()).min(Comparator.naturalOrder()).orElse(null)
    }

    /**
     * Use fitness-proportionate selection to obtain an [Individual] from the population.
     * @return An [Individual] from the population.
     */
    private fun roulette(): Individual {
        // calculate the total weight
        var weightSum = 0.0
        for (c in population) {
            val v = 1 - c.fitness
            weightSum += v
        }

        // Generate a random number between 0 and weightSum
        var rand = weightSum * Parameters.random.nextDouble()
        // Find random value based on weights
        for (indiv in population) {
            rand -= 1 - indiv.fitness
            if (rand < 0) return indiv
        }
        // When rounding errors occur, return the last item
        return population[-1]
    }

    /**
     * Rank the routes and select the individual that meets the selection criteria.
     * @return An [Individual] from the population.
     */
    private fun rankRoutes(): Individual {
        val fitness = DoubleArray(Parameters.populationSize)
        for (i in 0 until Parameters.populationSize) {
            fitness[i] = (i + 1).toDouble()
        }
        Helpers.unitize1(fitness)
        return population[Helpers.random(fitness)]
    }
    /* ******************************* */ /* ********** CROSSOVER ********** */ /* ******************************* */
    /**
     * Use uniform crossover to create children from the population.
     * @param parent1 An [Individual] from the population.
     * @param parent2 An [Individual] from the population.
     * @return A collection of children ([Individual]s) from the population.
     */
    private fun uniform(parent1: Individual, parent2: Individual): java.util.ArrayList<Individual> {
        val child1 = Individual()
        val child2 = Individual()
        var i = 0
        if (i < parent1.chromosome.size) {
            do {
                if (Parameters.random.nextBoolean()) {
                    child1.chromosome[i] = parent1.chromosome[i]
                    child2.chromosome[i] = parent2.chromosome[i]
                } else {
                    child1.chromosome[i] = parent2.chromosome[i]
                    child2.chromosome[i] = parent1.chromosome[i]
                }
                i++
            } while (i < parent1.chromosome.size)
        }
        return java.util.ArrayList(listOf(child1, child2))
    }

    /**
     * Use One-Point crossover to create a new collection of children from the population.
     * @param parent1 An [Individual] from the population.
     * @param parent2 An [Individual] from the population.
     * @return A collection of children ([Individual]s) from the population.
     */
    private fun onePoint(parent1: Individual, parent2: Individual): java.util.ArrayList<Individual> {
        val child1 = Individual()
        val child2 = Individual()
        val cut = Parameters.random.nextInt(parent1.chromosome.size)
        IntStream.range(0, parent1.chromosome.size).forEach { i: Int ->
            if (i < cut) {
                child1.chromosome[i] = parent1.chromosome[i]
                child2.chromosome[i] = parent2.chromosome[i]
            } else {
                child1.chromosome[i] = parent2.chromosome[i]
                child2.chromosome[i] = parent1.chromosome[i]
            }
        }
        return java.util.ArrayList(listOf(child1, child2))
    }

    /**
     * Use Two-Point crossover to create a new collection of children from the population.
     * @param parent1 An [Individual] from the population.
     * @param parent2 An [Individual] from the population.
     * @return A collection of children ([Individual]s) from the population.
     */
    private fun twoPoint(parent1: Individual, parent2: Individual): java.util.ArrayList<Individual> {
        val child1 = Individual()
        val child2 = Individual()
        val chromosomeLength = parent1.chromosome.size
        val cutPoint1 = Parameters.random.nextInt(chromosomeLength)
        val cutPoint2 = Parameters.random.nextInt(chromosomeLength - cutPoint1 + 1) + cutPoint1
        IntStream.range(0, chromosomeLength).forEach { i: Int ->
            if (i < cutPoint1 || i >= cutPoint2) {
                child1.chromosome[i] = parent1.chromosome[i]
                child2.chromosome[i] = parent2.chromosome[i]
            } else {
                child1.chromosome[i] = parent2.chromosome[i]
                child2.chromosome[i] = parent1.chromosome[i]
            }
        }
        return java.util.ArrayList(listOf(child1, child2))
    }

    /**
     * Use arithmetic crossover to create a new collection of children from the population.
     * @param parent1 An [Individual] from the population.
     * @param parent2 An [Individual] from the population.
     * @return A collection of children ([Individual]s) from the population.
     */
    private fun arithmetic(parent1: Individual, parent2: Individual): java.util.ArrayList<Individual> {
        val child = Individual()
        IntStream.range(0, parent1.chromosome.size).forEach { i: Int ->
            val average = (parent1.chromosome[i] + parent2.chromosome[i]) / 2
            child.chromosome[i] = average
        }
        return java.util.ArrayList(listOf(child))
    }
    /* ******************************* */ /* *********** MUTATION ********** */ /* ******************************* */
    /**
     * Use standard mutation to mutate a chromosome based on a predetermined mutation rate.
     * @param individuals The population of [Individual]s.
     */
    private fun mutate(individuals: java.util.ArrayList<Individual>) {
        for (individual in individuals) {
            IntStream.range(0, individual.chromosome.size)
                .filter { Parameters.random.nextDouble() < Parameters.mutateRate }
                .forEachOrdered { i: Int ->
                    if (Parameters.random.nextBoolean()) {
                        individual.chromosome[i] += Parameters.mutateChange
                    } else {
                        individual.chromosome[i] -= Parameters.mutateChange
                    }
                }
        }
    }

    /**
     * Use constrained mutation to mutate a chromosome within a poplulation.
     * @param individuals The population of [Individual]s.
     */
    private fun constrained(individuals: java.util.ArrayList<Individual>) {
        for (individual in individuals) {
            IntStream.range(0, individual.chromosome.size)
                .filter { Parameters.random.nextDouble() < Parameters.mutateRate }
                .forEachOrdered { i: Int ->
                    val prior = individual.fitness
                    if (Parameters.random.nextBoolean()) {
                        individual.chromosome[i] += Parameters.mutateChange
                        individual.fitness = Fitness.evaluate(individual, this)
                        if (individual.fitness > prior) individual.chromosome[i] -= Parameters.mutateChange // Undo if bad.
                    } else {
                        individual.chromosome[i] -= Parameters.mutateChange
                        individual.fitness = Fitness.evaluate(individual, this)
                        if (individual.fitness > prior) individual.chromosome[i] += Parameters.mutateChange // Undo if bad.
                    }
                }
        }
    }

    /**
     * Taking inspiration from the [SimulatedAnnealing] class, this mutation operator uses
     * concepts similar to Simulated Annealing to achieve mutation.
     * @param individuals A population of [Individual]s.
     * @param temperature The temperature.
     */
    private fun annealing(individuals: java.util.ArrayList<Individual>, temperature: Double) {
        for ((index, individual) in individuals.withIndex()) {
            val i = individual.copy()

            // Get a random genes in the chromosome (change with next int)
            val pos1 = (i.chromosome.size * Parameters.random.nextDouble()).toInt()
            val pos2 = (i.chromosome.size * Parameters.random.nextDouble()).toInt()

            // Get the values at selected positions in the chromosome
            val swap1 = i.chromosome[pos1]
            val swap2 = i.chromosome[pos2]

            // Swap them
            i.chromosome[pos1] = swap2
            i.chromosome[pos2] = swap1

            // Evaluate fitness
            i.fitness = Fitness.evaluate(i, this)

            // Get energy of solutions
            val current = individual.fitness
            val neighbour = i.fitness

            // Decide if we should accept the neighbour
            if (Helpers.acceptance(current, neighbour, temperature)
                >= Parameters.random.nextDouble()
            ) individuals[index] = i
        }
    }

    /**
     * Replace the worst [Individual]s in the population with new random [Individual]s
     * @param individuals A collection of [Individual]s from the population.
     */
    private fun worst(individuals: java.util.ArrayList<Individual>) {
        individuals.forEach(Consumer { individual: Individual? ->
            val index = worst
            population[index] = individual
        })
    }

    /**
     * Similar to selection, but replace the worst [Individual]s in the population.
     * @param individuals A collection of [Individual]s
     */
    private fun tournament(individuals: java.util.ArrayList<Individual>) {
        val tournament = Parameters.TOURNAMENT_SIZE
        individuals.forEach(Consumer { individual: Individual? ->
            population.shuffle()
            val worst = population.stream().limit(tournament.toLong()).max(Comparator.naturalOrder()).orElse(null)
            population.remove(worst)
            population.add(individual)
        })
    }

    /**
     * Obtain the index of the worst performing [Individual] in the population.
     * @return The index of the worst performing [Individual]s
     */
    private val worst: Int
        get() {
            var worst: Individual? = null
            var index = -1
            for (i in population.indices) {
                val individual = population[i]
                if (worst == null || individual.fitness > worst.fitness) {
                    worst = individual
                    index = i
                }
            }
            return index
        }
}