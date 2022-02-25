package coursework.utility;

import coursework.Parameters;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

/**
 * This class contains several helper functions that help in the creation of the Evolutionary Algorithm
 * that is contained in the {@link coursework.ExampleEvolutionaryAlgorithm} class.
 */
public class Helpers {

    public static double activationFunction(double v) {
        final double max = Math.max(0, Math.min(1, (v + 1) / 2));
        return switch (Parameters.ACTIVATION) {
            case HARD_ELISH -> (v < 0) ? max * (Math.exp(v) - 1) : v * max;
            case LEAKY_RELU -> (v > 0) ? v : (v / 100);
            case RELU -> (v > 0) ? v : -1;
            case SELU -> (v > 0) ? v * 1.0507009 : 1.0507009 * (1.673263 * Math.exp(v)) - 1.673263;
            case STEP -> (v <= 0) ? -1.0d : 1.0d;
            case SWISH -> (v * (1 / (1 + Math.exp(-v))));
            case TANH -> (v < -20.0d) ? -1.0d : (v > 20.0d) ? 1.0d : Math.tanh(v);
            case ELU -> (v > 0) ? v : (Math.exp(v) - 1) / 10;
        };
    }

    /**
     * The acceptance rate for the Annealing operator.
     * @param currentFitness The current fitness
     * @param newFitness The new fitness
     * @param temperature The temperature.
     * @return 1 if the current fitness is less than the new fitness, exp(current - new)/temperature otherwise.
     */
    public static double acceptance(double currentFitness, double newFitness, double temperature) {
        return currentFitness < newFitness ? 1.0 : Math.exp((currentFitness - newFitness) / temperature);
    }

    @Contract(pure = true)
    private static double norm1(double @NotNull [] x) {
        double norm = 0.0;

        for (double n : x) {
            norm += Math.abs(n);
        }

        return norm;
    }

    public static void unitize1(double[] x) {
        double n = norm1(x);
        for (int i = 0; i < x.length; i++) {
            x[i] /= n;
        }
    }

    public static int random(double[] prob) {
        int[] ans = random(prob, 1);
        return ans[0];
    }

    private static int @NotNull [] random(double @NotNull [] prob, int n) {
        // set up alias table
        double[] q = new double[prob.length];
        for (int i = 0; i < prob.length; i++) {
            q[i] = prob[i] * prob.length;
        }

        // initialize a with indices
        int[] a = new int[prob.length];
        for (int i = 0; i < prob.length; i++) {
            a[i] = i;
        }

        // set up H and L
        int[] HL = new int[prob.length];
        int head = 0;
        int tail = prob.length - 1;
        for (int i = 0; i < prob.length; i++) {
            if (q[i] >= 1.0) {
                HL[head++] = i;
            } else {
                HL[tail--] = i;
            }
        }

        while (head != 0 && tail != prob.length - 1) {
            int j = HL[tail + 1];
            int k = HL[head - 1];
            a[j] = k;
            q[k] += q[j] - 1;
            tail++;                                  // remove j from L
            if (q[k] < 1.0) {
                HL[tail--] = k;                      // add k to L
                head--;                              // remove k
            }
        }

        // generate sample
        int[] ans = new int[n];
        for (int i = 0; i < n; i++) {
            double rU = Parameters.random.nextDouble() * prob.length;

            int k = (int) (rU);
            rU -= k;  /* rU becomes rU-[rU] */

            if (rU < q[k]) {
                ans[i] = k;
            } else {
                ans[i] = a[k];
            }
        }

        return ans;
    }
}
