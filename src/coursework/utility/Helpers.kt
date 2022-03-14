package coursework.utility

import coursework.Parameters
import coursework.options.Activation
import org.jetbrains.annotations.Contract
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.tanh

/**
 * This class contains several helper functions that help in the creation of the Evolutionary Algorithm
 * that is contained in the [coursework.ExampleEvolutionaryAlgorithm] class.
 */
object Helpers {

    /**
     * A list of the possible activation functions that can be used within the program.
     */
    fun activationFunction(v: Double): Double {
        return when (Parameters.ACTIVATION) {
            Activation.LEAKY_RELU -> if (v > 0) v else v / 100
            Activation.RELU -> if (v > 0.0) v else -1.0
            Activation.STEP -> if (v <= 0) -1.0 else 1.0
            Activation.SWISH -> v * (1 / (1 + exp(-v)))
            Activation.TANH -> if (v < -20.0) -1.0 else if (v > 20.0) 1.0 else tanh(v)
            Activation.ELU -> if (v > 0) v else (exp(v) - 1) / 10
            else -> throw IllegalArgumentException()
        }
    }

    /**
     * The acceptance rate for the Annealing operator.
     * @param currentFitness The current fitness
     * @param newFitness The new fitness
     * @param temperature The temperature.
     * @return 1 if the current fitness is less than the new fitness, exp(current - new)/temperature otherwise.
     */
    fun acceptance(currentFitness: Double, newFitness: Double, temperature: Double): Double {
        return if (currentFitness < newFitness) 1.0 else exp((currentFitness - newFitness) / temperature)
    }

    @Contract(pure = true)
    private fun norm(x: DoubleArray): Double {
        var norm = 0.0
        for (n in x) {
            norm += abs(n)
        }
        return norm
    }

    fun unitise(x: DoubleArray) {
        val n = norm(x)
        for (i in x.indices) {
            x[i] /= n
        }
    }

    fun random(prob: DoubleArray): Int {
        val ans = random(prob, 1)
        return ans[0]
    }

    private fun random(prob: DoubleArray, n: Int): IntArray {
        // set up alias table
        val q = DoubleArray(prob.size)
        for (i in prob.indices) {
            q[i] = prob[i] * prob.size
        }

        // initialize a with indices
        val a = IntArray(prob.size)
        for (i in prob.indices) {
            a[i] = i
        }

        // set up H and L
        val HL = IntArray(prob.size)
        var head = 0
        var tail = prob.size - 1
        for (i in prob.indices) {
            if (q[i] >= 1.0) {
                HL[head++] = i
            } else {
                HL[tail--] = i
            }
        }
        while (head != 0 && tail != prob.size - 1) {
            val j = HL[tail + 1]
            val k = HL[head - 1]
            a[j] = k
            q[k] += q[j] - 1
            tail++ // remove j from L
            if (q[k] < 1.0) {
                HL[tail--] = k // add k to L
                head-- // remove k
            }
        }

        // generate sample
        val ans = IntArray(n)
        for (i in 0 until n) {
            var rU = Parameters.random.nextDouble() * prob.size
            val k = rU.toInt()
            rU -= k.toDouble() /* rU becomes rU-[rU] */
            if (rU < q[k]) {
                ans[i] = k
            } else {
                ans[i] = a[k]
            }
        }
        return ans
    }
}
