package coursework

import model.NeuralNetwork
import model.Individual
import model.Fitness
import kotlin.jvm.JvmStatic
import kotlin.math.tanh

class ExampleHillClimber : NeuralNetwork() {
    override fun run() {
        //initialise a single individual
        best = Individual()

        //run for max evaluations
        for (gen in 0 until Parameters.maxEvaluations) {
            //mutate the best
            val candidate = mutateBest()

            //accept if better
            if (candidate.fitness < best.fitness) {
                best = candidate
            }
            outputStats()
        }
        saveNeuralNetwork()
    }

    private fun mutateBest(): Individual {
        val candidate = best.copy()
        for (i in candidate.chromosome.indices) {
            if (Parameters.random.nextDouble() < Parameters.mutateRate) {
                if (Parameters.random.nextBoolean()) {
                    candidate.chromosome[i] += Parameters.mutateChange
                } else {
                    candidate.chromosome[i] -= Parameters.mutateChange
                }
            }
        }
        Fitness.evaluate(candidate, this)
        return candidate
    }

    override fun activationFunction(x: Double): Double {
        if (x < -20.0) {
            return -1.0
        } else if (x > 20.0) {
            return 1.0
        }
        return tanh(x)
    }

    companion object {
        @JvmStatic
        fun main(args: Array<String>) {
            val hillClimber: NeuralNetwork = ExampleHillClimber()
            hillClimber.run()
        }
    }
}