package coursework

import kotlin.jvm.JvmStatic
import coursework.StartNoGui
import java.io.IOException
import kotlin.Throws
import model.LunarParameters.DataSet
import model.NeuralNetwork
import coursework.ExampleEvolutionaryAlgorithm
import model.Fitness
import java.io.FileWriter
import coursework.options.Initialisation
import coursework.options.Selection
import coursework.options.Crossover
import coursework.options.Replacement
import coursework.options.Activation
import coursework.SimulatedAnnealing
import coursework.options.Mutation
import org.jetbrains.annotations.Contract

/**
 * Example of how to run the [ExampleEvolutionaryAlgorithm] without the need for the GUI
 * This allows you to conduct multiple runs programmatically
 * The code runs faster when not required to update a user interface
 */
object StartNoGui {
    @JvmStatic
    fun main(args: Array<String>) {
        /* Train the Neural Network using our Evolutionary Algorithm */

        /*
         * Or We can reload the NN from the file generated during training and test it on a data set
         * We can supply a filename or null to open a file dialog
         * Note that files must be in the project root and must be named *-n.txt
         * where "n" is the number of hidden nodes
         * ie  1518461386696-5.txt was saved at timestamp 1518461386696 and has 5 hidden nodes
         * Files are saved automatically at the end of training
         *
         *  Uncomment the following code and replace the name of the saved file to test a previously trained network
         */

//		NeuralNetwork nn2 = NeuralNetwork.loadNeuralNetwork("1234567890123-5.txt");
//		Parameters.setDataSet(DataSet.Random);
//		double fitness2 = Fitness.evaluate(nn2);
//		System.out.println("Fitness on " + Parameters.getDataSet() + " " + fitness2);
//
        try {
            findBestPopulationSize()
            findBestMutationRate()
            findBestMutationChange()
            findBestHiddenNodes()
            findBestSelection()
            testTournamentSize()
            findBestCrossover()
            findBestInitialisation()
            findBestReplace()
            findBestActivation()
            findBestMutation()
            findMinMaxGenes()
            findBestSACoolRate()
            findBestActivationSA()
            testAlgorithms()
        } catch (e: IOException) {
            // TODO Auto-generated catch block
            e.printStackTrace()
        }
    }

    @Contract("_ -> new")
    private fun convertToCSV(data: Array<String>): String {
        return java.lang.String.join(",", *data)
    }

    @Throws(IOException::class)
    private fun findBestPopulationSize() {
        var i = 10
        while (i <= 110) {
            if (i == 60) {
                i += 10
                continue
            }
            // For each pop size
            var avgTrain = 0.0
            var avgTest = 0.0
            val runs = 10
            for (r in 0 until runs) {
                Parameters.populationSize = i
                //Set the data set for training 
                Parameters.setDataSet(DataSet.Training)

                //Create a new Neural Network Trainer Using the above parameters 
                val nn: NeuralNetwork = ExampleEvolutionaryAlgorithm()
                nn.run()

                // train
                val trainFitness = Fitness.evaluate(nn)
                // test
                Parameters.setDataSet(DataSet.Test)
                val testFitness = Fitness.evaluate(nn)
                avgTrain += trainFitness
                avgTest += testFitness
            }

            // once r runs have completed
            val dataLines = arrayOf(
                "" + Parameters.populationSize,
                String.format("%.5f", avgTrain / runs),
                String.format("%.5f", avgTest / runs)
            )
            val line = convertToCSV(dataLines)
            val csvOutputFile = FileWriter("results/results.csv", true)
            csvOutputFile.write(
                """
    $line
    
    """.trimIndent()
            ) //appends the string to the file
            csvOutputFile.close()
            i += 10
        }
    }

    @Throws(IOException::class)
    private fun findBestMutationRate() {
        var i = 0.45
        while (i < 1.05) {

            // For each pop size
            var avgTrain = 0.0
            var avgTest = 0.0
            val runs = 10
            for (r in 0 until runs) {
                Parameters.mutateRate = i
                //Set the data set for training 
                Parameters.setDataSet(DataSet.Training)

                //Create a new Neural Network Trainer Using the above parameters 
                val nn: NeuralNetwork = ExampleEvolutionaryAlgorithm()
                nn.run()

                // train
                val trainFitness = Fitness.evaluate(nn)
                // test
                Parameters.setDataSet(DataSet.Test)
                val testFitness = Fitness.evaluate(nn)
                avgTrain += trainFitness
                avgTest += testFitness
            }

            // once r runs have completed
            val dataLines = arrayOf(
                "" + Parameters.mutateRate,
                String.format("%.5f", avgTrain / runs),
                String.format("%.5f", avgTest / runs)
            )
            val line = convertToCSV(dataLines)
            val csvOutputFile = FileWriter("results/results.csv", true)
            csvOutputFile.write(
                """
    $line
    
    """.trimIndent()
            ) //appends the string to the file
            csvOutputFile.close()
            i += 0.10
        }
    }

    @Throws(IOException::class)
    private fun findBestMutationChange() {
        var i = 1.15
        while (i < 1.75) {

            // For each pop size
            var avgTrain = 0.0
            var avgTest = 0.0
            val runs = 10
            for (r in 0 until runs) {
                Parameters.mutateChange = i
                //Set the data set for training 
                Parameters.setDataSet(DataSet.Training)

                //Create a new Neural Network Trainer Using the above parameters 
                val nn: NeuralNetwork = ExampleEvolutionaryAlgorithm()
                nn.run()

                // train
                val trainFitness = Fitness.evaluate(nn)
                // test
                Parameters.setDataSet(DataSet.Test)
                val testFitness = Fitness.evaluate(nn)
                avgTrain += trainFitness
                avgTest += testFitness
            }

            // once r runs have completed
            val dataLines = arrayOf(
                "" + Parameters.mutateChange,
                String.format("%.5f", avgTrain / runs),
                String.format("%.5f", avgTest / runs)
            )
            val line = convertToCSV(dataLines)
            val csvOutputFile = FileWriter("results/results.csv", true)
            csvOutputFile.write(
                """
    $line
    
    """.trimIndent()
            ) //appends the string to the file
            csvOutputFile.close()
            i += 0.10
        }
    }

    @Throws(IOException::class)
    private fun findBestHiddenNodes() {
        for (i in 3..14) {
            // For each pop size
            var avgTrain = 0.0
            var avgTest = 0.0
            val runs = 10
            for (r in 0 until runs) {
                //Set the data set for training 
                Parameters.setDataSet(DataSet.Training)
                Parameters.setHidden(i)

                //Create a new Neural Network Trainer Using the above parameters 
                val nn: NeuralNetwork = ExampleEvolutionaryAlgorithm()
                nn.run()

                // train
                val trainFitness = Fitness.evaluate(nn)
                // test
                Parameters.setDataSet(DataSet.Test)
                val testFitness = Fitness.evaluate(nn)
                avgTrain += trainFitness
                avgTest += testFitness
            }
            // once r runs have completed
            val dataLines = arrayOf(
                "" + Parameters.getNumHidden(),
                String.format("%.5f", avgTrain / runs),
                String.format("%.5f", avgTest / runs)
            )
            val line = convertToCSV(dataLines)
            val csvOutputFile = FileWriter("results/results.csv", true)
            csvOutputFile.write(
                """
    $line
    
    """.trimIndent()
            ) //appends the string to the file
            csvOutputFile.close()
        }
    }

    @Throws(IOException::class)
    private fun findBestInitialisation() {
        for (initialisationType in Initialisation.values()) {
            // For each pop size
            var avgTrain = 0.0
            var avgTest = 0.0
            val runs = 20
            for (r in 0 until runs) {
                //Set the data set for training 
                Parameters.setDataSet(DataSet.Training)
                Parameters.INITIALISATION = initialisationType

                //Create a new Neural Network Trainer Using the above parameters 
                val nn: NeuralNetwork = ExampleEvolutionaryAlgorithm()
                nn.run()

                // train
                val trainFitness = Fitness.evaluate(nn)
                // test
                Parameters.setDataSet(DataSet.Test)
                val testFitness = Fitness.evaluate(nn)
                avgTrain += trainFitness
                avgTest += testFitness
            }
            // once r runs have completed
            val dataLines = arrayOf(
                "" + Parameters.INITIALISATION,
                String.format("%.5f", avgTrain / runs),
                String.format("%.5f", avgTest / runs)
            )
            val line = convertToCSV(dataLines)
            val csvOutputFile = FileWriter("results/results.csv", true)
            csvOutputFile.write(
                """
    $line
    
    """.trimIndent()
            ) //appends the string to the file
            csvOutputFile.close()
        }
    }

    @Throws(IOException::class)
    private fun findBestSelection() {
        for (selectionType in Selection.values()) {
            // For each pop size
            var avgTrain = 0.0
            var avgTest = 0.0
            val runs = 20
            for (r in 0 until runs) {
                //Set the data set for training 
                Parameters.setDataSet(DataSet.Training)
                Parameters.SELECTION = selectionType

                //Create a new Neural Network Trainer Using the above parameters 
                val nn: NeuralNetwork = ExampleEvolutionaryAlgorithm()
                nn.run()

                // train
                val trainFitness = Fitness.evaluate(nn)
                // test
                Parameters.setDataSet(DataSet.Test)
                val testFitness = Fitness.evaluate(nn)
                avgTrain += trainFitness
                avgTest += testFitness
            }
            // once r runs have completed
            val dataLines = arrayOf(
                "" + Parameters.SELECTION, String.format("%.5f", avgTrain / runs), String.format("%.5f", avgTest / runs)
            )
            val line = convertToCSV(dataLines)
            val csvOutputFile = FileWriter("results/results.csv", true)
            csvOutputFile.write(
                """
    $line
    
    """.trimIndent()
            ) //appends the string to the file
            csvOutputFile.close()
        }
    }

    @Throws(IOException::class)
    private fun findBestCrossover() {
        for (crossoverType in Crossover.values()) {
            // For each pop size
            var avgTrain = 0.0
            var avgTest = 0.0
            val runs = 20
            for (r in 0 until runs) {
                //Set the data set for training 
                Parameters.setDataSet(DataSet.Training)
                Parameters.CROSSOVER = crossoverType

                //Create a new Neural Network Trainer Using the above parameters 
                val nn: NeuralNetwork = ExampleEvolutionaryAlgorithm()
                nn.run()

                // train
                val trainFitness = Fitness.evaluate(nn)
                // test
                Parameters.setDataSet(DataSet.Test)
                val testFitness = Fitness.evaluate(nn)
                avgTrain += trainFitness
                avgTest += testFitness
            }
            // once r runs have completed
            val dataLines = arrayOf(
                "" + Parameters.CROSSOVER, String.format("%.5f", avgTrain / runs), String.format("%.5f", avgTest / runs)
            )
            val line = convertToCSV(dataLines)
            val csvOutputFile = FileWriter("results/results.csv", true)
            csvOutputFile.write(
                """
    $line
    
    """.trimIndent()
            ) //appends the string to the file
            csvOutputFile.close()
        }
    }

    @Throws(IOException::class)
    private fun findBestReplace() {
        for (replaceType in Replacement.values()) {
            // For each pop size
            var avgTrain = 0.0
            var avgTest = 0.0
            val runs = 20
            for (r in 0 until runs) {
                //Set the data set for training 
                Parameters.setDataSet(DataSet.Training)
                Parameters.REPLACEMENT = replaceType

                //Create a new Neural Network Trainer Using the above parameters 
                val nn: NeuralNetwork = ExampleEvolutionaryAlgorithm()
                nn.run()

                // train
                val trainFitness = Fitness.evaluate(nn)
                // test
                Parameters.setDataSet(DataSet.Test)
                val testFitness = Fitness.evaluate(nn)
                avgTrain += trainFitness
                avgTest += testFitness
            }
            // once r runs have completed
            val dataLines = arrayOf(
                "" + Parameters.REPLACEMENT,
                String.format("%.5f", avgTrain / runs),
                String.format("%.5f", avgTest / runs)
            )
            val line = convertToCSV(dataLines)
            val csvOutputFile = FileWriter("results/results.csv", true)
            csvOutputFile.write(
                """
    $line
    
    """.trimIndent()
            ) //appends the string to the file
            csvOutputFile.close()
        }
    }

    @Throws(IOException::class)
    private fun testTournamentSize() {
        val sizesToTest = intArrayOf(5, 10, 20, 30, 50, 70, 90)
        for (i in sizesToTest) {
            // For each pop size
            var avgTrain = 0.0
            var avgTest = 0.0
            val runs = 20
            for (r in 0 until runs) {
                //Set the data set for training 
                Parameters.setDataSet(DataSet.Training)
                Parameters.SELECTION = Selection.TOURNAMENT
                Parameters.TOURNAMENT_SIZE = i

                //Create a new Neural Network Trainer Using the above parameters 
                val nn: NeuralNetwork = ExampleEvolutionaryAlgorithm()
                nn.run()

                // train
                val trainFitness = Fitness.evaluate(nn)
                // test
                Parameters.setDataSet(DataSet.Test)
                val testFitness = Fitness.evaluate(nn)
                avgTrain += trainFitness
                avgTest += testFitness
            }
            // once r runs have completed
            val dataLines = arrayOf(
                "" + Parameters.TOURNAMENT_SIZE,
                String.format("%.5f", avgTrain / runs),
                String.format("%.5f", avgTest / runs)
            )
            val line = convertToCSV(dataLines)
            val csvOutputFile = FileWriter("results/results.csv", true)
            csvOutputFile.write(
                """
    $line
    
    """.trimIndent()
            ) //appends the string to the file
            csvOutputFile.close()
        }
    }

    @Throws(IOException::class)
    private fun findMinMaxGenes() {
        var i = 0.1
        while (i < 1) {

            // For each pop size
            var avgTrain = 0.0
            var avgTest = 0.0
            val runs = 10
            for (r in 0 until runs) {
                //Set the data set for training 
                Parameters.setDataSet(DataSet.Training)
                Parameters.minGene = -i
                Parameters.maxGene = i

                //Create a new Neural Network Trainer Using the above parameters 
                val nn: NeuralNetwork = ExampleEvolutionaryAlgorithm()
                nn.run()

                // train
                val trainFitness = Fitness.evaluate(nn)
                // test
                Parameters.setDataSet(DataSet.Test)
                val testFitness = Fitness.evaluate(nn)
                avgTrain += trainFitness
                avgTest += testFitness
            }
            // once r runs have completed
            val dataLines = arrayOf(
                "" + Parameters.minGene,
                "" + Parameters.maxGene, String.format("%.5f", avgTrain / runs), String.format("%.5f", avgTest / runs)
            )
            val line = convertToCSV(dataLines)
            val csvOutputFile = FileWriter("results/results.csv", true)
            csvOutputFile.write(
                """
    $line
    
    """.trimIndent()
            ) //appends the string to the file
            csvOutputFile.close()
            i += 0.1
        }
    }

    @Throws(IOException::class)
    private fun findBestActivation() {
        for (activationType in Activation.values()) {
            // For each pop size
            var avgTrain = 0.0
            var avgTest = 0.0
            val runs = 10
            for (r in 0 until runs) {
                //Set the data set for training 
                Parameters.setDataSet(DataSet.Training)
                Parameters.ACTIVATION = activationType

                //Create a new Neural Network Trainer Using the above parameters 
                val nn: NeuralNetwork = ExampleEvolutionaryAlgorithm()
                nn.run()

                // train
                val trainFitness = Fitness.evaluate(nn)
                // test
                Parameters.setDataSet(DataSet.Test)
                val testFitness = Fitness.evaluate(nn)
                avgTrain += trainFitness
                avgTest += testFitness
            }
            // once r runs have completed
            val dataLines = arrayOf(
                "" + Parameters.ACTIVATION,
                String.format("%.5f", avgTrain / runs),
                String.format("%.5f", avgTest / runs)
            )
            val line = convertToCSV(dataLines)
            val csvOutputFile = FileWriter("results/results.csv", true)
            csvOutputFile.write(
                """
    $line
    
    """.trimIndent()
            ) //appends the string to the file
            csvOutputFile.close()
        }
    }

    @Throws(IOException::class)
    private fun findBestSACoolRate() {
        var i = 0.0007
        while (i < 0.0013) {

            // For each pop size
            var avgTrain = 0.0
            var avgTest = 0.0
            val runs = 10
            for (r in 0 until runs) {
                //Set the data set for training 
                Parameters.setDataSet(DataSet.Training)
                Parameters.COOLING_RATE = i

                //Create a new Neural Network Trainer Using the above parameters 
                val nn: NeuralNetwork = ExampleEvolutionaryAlgorithm()
                nn.run()

                // train
                val trainFitness = Fitness.evaluate(nn)
                // test
                Parameters.setDataSet(DataSet.Test)
                val testFitness = Fitness.evaluate(nn)
                avgTrain += trainFitness
                avgTest += testFitness
            }
            // once r runs have completed
            val dataLines = arrayOf(
                "" + Parameters.COOLING_RATE,
                String.format("%.5f", avgTrain / runs),
                String.format("%.5f", avgTest / runs)
            )
            val line = convertToCSV(dataLines)
            val csvOutputFile = FileWriter("results/results.csv", true)
            csvOutputFile.write(
                """
    $line
    
    """.trimIndent()
            ) //appends the string to the file
            csvOutputFile.close()
            i += 0.0001
        }
    }

    @Throws(IOException::class)
    private fun findBestActivationSA() {
        for (activationType in Activation.values()) {
            // For each pop size
            var avgTrain = 0.0
            var avgTest = 0.0
            val runs = 5
            for (r in 0 until runs) {
                //Set the data set for training 
                Parameters.setDataSet(DataSet.Training)
                Parameters.ACTIVATION = activationType

                //Create a new Neural Network Trainer Using the above parameters 
                val nn: NeuralNetwork = SimulatedAnnealing()
                nn.run()

                // train
                val trainFitness = Fitness.evaluate(nn)
                // test
                Parameters.setDataSet(DataSet.Test)
                val testFitness = Fitness.evaluate(nn)
                avgTrain += trainFitness
                avgTest += testFitness
            }
            // once r runs have completed
            val dataLines = arrayOf(
                "" + Parameters.ACTIVATION,
                String.format("%.5f", avgTrain / runs),
                String.format("%.5f", avgTest / runs)
            )
            val line = convertToCSV(dataLines)
            val csvOutputFile = FileWriter("results/results.csv", true)
            csvOutputFile.write(
                """
    $line
    
    """.trimIndent()
            ) //appends the string to the file
            csvOutputFile.close()
        }
    }

    @Throws(IOException::class)
    private fun findBestMutation() {
        for (mutationType in Mutation.values()) {
            // For each pop size
            var avgTrain = 0.0
            var avgTest = 0.0
            val runs = 10
            for (r in 0 until runs) {
                //Set the data set for training 
                Parameters.setDataSet(DataSet.Training)
                Parameters.MUTATION = mutationType

                //Create a new Neural Network Trainer Using the above parameters 
                val nn: NeuralNetwork = ExampleEvolutionaryAlgorithm()
                nn.run()

                // train
                val trainFitness = Fitness.evaluate(nn)
                // test
                Parameters.setDataSet(DataSet.Test)
                val testFitness = Fitness.evaluate(nn)
                avgTrain += trainFitness
                avgTest += testFitness
            }
            // once r runs have completed
            val dataLines = arrayOf(
                "" + Parameters.MUTATION, String.format("%.5f", avgTrain / runs), String.format("%.5f", avgTest / runs)
            )
            val line = convertToCSV(dataLines)
            val csvOutputFile = FileWriter("results/results.csv", true)
            csvOutputFile.write(
                """
    $line
    
    """.trimIndent()
            ) //appends the string to the file
            csvOutputFile.close()
        }
    }

    @Throws(IOException::class)
    private fun testAlgorithms() {
        for (i in 0..1) {
            // For each pop size
            var avgTrain = 0.0
            var avgTest = 0.0
            val runs = 5
            var network = ""
            for (r in 0 until runs) {
                //Set the data set for training 
                Parameters.setDataSet(DataSet.Training)

                //Create a new Neural Network Trainer Using the above parameters
                var nn: NeuralNetwork
                if (i == 0) {
                    nn = ExampleEvolutionaryAlgorithm()
                    network = "SS-GA"
                } else {
                    nn = SimulatedAnnealing()
                    network = "SA"
                }
                nn.run()

                // train
                val trainFitness = Fitness.evaluate(nn)
                // test
                Parameters.setDataSet(DataSet.Test)
                val testFitness = Fitness.evaluate(nn)
                avgTrain += trainFitness
                avgTest += testFitness
            }
            // once r runs have completed
            val dataLines = arrayOf(
                network, String.format("%.5f", avgTrain / runs), String.format("%.5f", avgTest / runs)
            )
            val line = convertToCSV(dataLines)
            val csvOutputFile = FileWriter("results/results.csv", true)
            csvOutputFile.write(
                """
    $line
    
    """.trimIndent()
            ) //appends the string to the file
            csvOutputFile.close()
        }
    }
}