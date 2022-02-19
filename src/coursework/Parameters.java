package coursework;

import model.LunarParameters;
import model.LunarParameters.DataSet;
import model.NeuralNetwork;

import java.lang.reflect.Field;
import java.util.Random;

public class Parameters {
	/**
	 * These parameter values can be changed
	 * You may add other Parameters as required to this class
	 */
	private static int numHidden = 5;
	private static int numGenes = calculateNumGenes();
	public static double minGene = -3; // specifies minimum and maximum weight values 
	//set the NeuralNetwork class here to use your code from the GUI
	public static final Class<? extends NeuralNetwork> neuralNetworkClass = ExampleEvolutionaryAlgorithm.class;

	public static int popSize = 40;
	public static int maxEvaluations = 20000;

	// Parameters for mutation 
	// Rate = probability of changing a gene
	// Change = the maximum +/- adjustment to the gene value
	public static double mutateRate = 0.04; // mutation rate for mutation operator
	public static double mutateChange = 0.1; // delta change for mutation operator
	
	//Random number generator used throughout the application
	public static long seed = System.currentTimeMillis();
	public static Random random = new Random(seed);
	public static double maxGene = 3;
	
	/**
	 * Do not change any methods that appear below here.
	 * 
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

	public static String printParams() {
		StringBuilder str = new StringBuilder();
		for(Field field : Parameters.class.getDeclaredFields()) {
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
	
	public static void setDataSet(DataSet dataSet) {
		LunarParameters.setDataSet(dataSet);
	}
	
	public static DataSet getDataSet() {
		return LunarParameters.getDataSet();
	}
	
	public static void main(String[] args) {
		printParams();
	}
}
