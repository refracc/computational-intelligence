package coursework;

import coursework.options.*;
import model.LunarParameters;
import model.LunarParameters.DataSet;
import model.NeuralNetwork;
import org.jetbrains.annotations.NotNull;

import java.lang.reflect.Field;
import java.util.Random;

public class Parameters {

    // Set the NeuralNetwork class here to use your code from the GUI
    public static final Class<? extends NeuralNetwork> neuralNetworkClass = ExampleEvolutionaryAlgorithm.class;
    public static final double TEMPERATURE = 100000d;

    /*
     * Custom parameters
     */
    public static Initialisation INITIALISATION = Initialisation.AUGMENTED;
    public static Selection SELECTION = Selection.TOURNAMENT;
    public static Crossover CROSSOVER = Crossover.ONE_POINT;
    public static Mutation MUTATION = Mutation.STANDARD;
    public static Replacement REPLACEMENT = Replacement.WORST;
    public static Activation ACTIVATION = Activation.TANH;
    public static double COOLING_RATE = 0.0012d;
    public static int TOURNAMENT_SIZE = 10;

    // specifies minimum and maximum weight values
    public static double maxGene = 0.7;
    public static double minGene = -0.7;
    public static int populationSize = 80;
    public static int maxEvaluations = 20000;

    // Parameters for mutation
    // Rate = probability of changing a gene
    // Change = the maximum +/- adjustment to the gene value
    public static double mutateRate = 0.90; // mutation rate for mutation operator
    public static double mutateChange = 1.50; // delta change for mutation operator

    //Random number generator used throughout the application
    public static long seed = System.currentTimeMillis();
    public static Random random = new Random(seed);
    private static int numHidden = 8;
    private static int numGenes = calculateNumGenes();

    /**
     * Do not change any methods that appear below here.
     */

    public static int getNumGenes() {
        return numGenes;
    }

    private static int calculateNumGenes() {
        return (NeuralNetwork.numInput * numHidden) + (numHidden * NeuralNetwork.numOutput) + numHidden + NeuralNetwork.numOutput;
    }

    public static int getNumHidden() {
        return numHidden;
    }

    public static void setHidden(int nHidden) {
        numHidden = nHidden;
        numGenes = calculateNumGenes();
    }

    public static @NotNull String printParams() {
        StringBuilder str = new StringBuilder();
        for (Field field : Parameters.class.getDeclaredFields()) {
            String name = field.getName();
            Object val = null;

            try {
                val = field.get(null);
            } catch (IllegalArgumentException | IllegalAccessException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
            str.append(name).append(" \t").append(val).append("\r\n");


        }

        return str.toString();
    }

    public static DataSet getDataSet() {
        return LunarParameters.getDataSet();
    }

    public static void setDataSet(DataSet dataSet) {
        LunarParameters.setDataSet(dataSet);
    }

    public static void main(String[] args) {
        printParams();
    }
}