package coursework

import coursework.utility.Helpers
import model.NeuralNetwork
import model.Individual
import model.Fitness
import kotlin.math.exp

/**
 * This class is an implementation of the Simulated Annealing algorithm that can be used to attempt to solve the problem of landing lunar landers.
 * By default, the [SimulatedAnnealing] class extends [NeuralNetwork] to confer to the constraints placed upon the [Parameters] class' `neuralNetworkClass` variable.
 *
 * @author Stewart A
 * @version 1.26
 * @since 19/02/2022
 */
class SimulatedAnnealing : NeuralNetwork() {

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
        var temperature = Parameters.TEMPERATURE
        val cooling = Parameters.COOLING_RATE

        // Initialise a new individual and
        // evaluate its fitness.
        var individual = Individual()
        individual.fitness = Fitness.evaluate(individual, this)

        // Set this new individual as
        // the best individual.
        best = individual.copy()

        // Do for all evaluations in the program.
        for (i in 0 until Parameters.maxEvaluations) {
            val newIndividual = individual.copy()

            // Obtain pseudo-random gene locations from within the chromosome.
            val position1 = Parameters.random.nextInt(newIndividual.chromosome.size)
            val position2 = Parameters.random.nextInt(newIndividual.chromosome.size)

            // Set up gene swaps
            val swap1 = newIndividual.chromosome[position1]
            val swap2 = newIndividual.chromosome[position2]

            // Perform the swap.
            newIndividual.chromosome[position1] = swap2
            newIndividual.chromosome[position2] = swap1

            // Evaluate its fitness.
            newIndividual.fitness = Fitness.evaluate(newIndividual, this)

            // Decide to accept neighbouring chromosome.
            val acceptance = acceptance(individual.fitness, newIndividual.fitness, temperature)
            individual = if (acceptance > Parameters.random.nextDouble()) newIndividual.copy() else individual

            // Replace the best solution that has been found thus far...
            best = if (individual.fitness < best.fitness) individual.copy() else best

            // Cool the algorithm down a little...
            temperature *= 1 - cooling
            println(i.toString() + "\t" + best)

            // Write the statistics to the screen
            outputStats()
        }
        // Save the Neural Network weights.
        saveNeuralNetwork()
    }

    /**
     * The acceptance rate
     *
     * @param currentFitness The fitness value of the current individual
     * @param newFitness     The fitness value of another individual
     * @param temperature    The temperature in the SimulatedAnnealing algorithm.
     * @return 1 if accepted.
     */
    private fun acceptance(currentFitness: Double, newFitness: Double, temperature: Double): Double {
        return if (newFitness < currentFitness) 1.0 else exp((currentFitness - newFitness) / temperature)
    }
}