

using Newtonsoft.Json;

namespace NN;

public class NeuralNetwork(int inputLayerSize)
{
    public double LearingRate { get; set; } = 0.01;

    public Layer InputLayer { get; set; } = new(inputLayerSize);

    public Layer OutputLayer
    {
        get
        {
            var res = InputLayer;
            while (res.NextLayer is not null)
            {
                res = res.NextLayer;
            }

            return res;
        }
    }

    public int OutputLayerSize = 0;

    public void Test(List<(double[] input, double[] result)> testData)
    {
        int correctGuesses = 0;
        foreach (var (input, result) in testData)
        {
            PropagateForward(input);
            var maxPred = OutputLayer.Perceptrons.MaxBy(p => p.Activation);
            var res = result.Max();
            if (Array.IndexOf(OutputLayer.Perceptrons, maxPred) == Array.IndexOf(result, res))
            {
                correctGuesses++;
            }
        }

        Console.WriteLine($"Guessed correctly {correctGuesses} examples. Mistakes {testData.Count - correctGuesses}");
    }

    public void Train(List<(double[] input, double[] result)> trainingData, int epochs)
    {
        for (int i = 0; i < epochs; i++)
        {
            foreach (var (input, result) in trainingData)
            {
                PropagateForward(input);
                PropagateBackward(result);
            }

            var data = trainingData.Last();
            var totalCost = Enumerable.Range(0, OutputLayer.Perceptrons.Length)
                .Select(i => Loss(OutputLayer.Perceptrons[i].Activation, data.result[i]))
                .ToArray()
                .Sum() / OutputLayer.Perceptrons.Length;

            Console.WriteLine($"Total cost: {totalCost} epoch: {i + 1}.");
        }
    }

    private static double Activation(double x)
    {
        // sigmoid
        return 1 / (1 + Math.Exp(-x));
    }

    private static double Loss(double predicted, double expected)
    {
        return Math.Pow(predicted - expected, 2);
    }

    private static double LossDerivative(double predictedDer, double expectedDer, int numberOfOutputs)
    {
        return 2 / (double)numberOfOutputs * (predictedDer - expectedDer);
    }

    private static double ActivationDerivative(double x)
    {
        var activation = Activation(x);
        return activation - Math.Pow(activation, 2);
    }

    private void PropagateForward(double[] input)
    {
        // Validate input
        if (input.Length != InputLayer.Perceptrons.Length)
        {
            throw new Exception("Invalid number of input values.");
        }

        // Set input layer with actual input
        for (int i = 0; i < InputLayer.Perceptrons.Length; i++)
        {
            InputLayer.Perceptrons[i].Activation = input[i];
        }

        // Forward propagation
        var currentLayer = InputLayer.NextLayer;
        while (currentLayer is not null)
        {
            foreach (var perceptron in currentLayer.Perceptrons)
            {
                double sum = DotProduct(perceptron.WeightsVector!, currentLayer.PrevousLayer!.Perceptrons);
                perceptron.Activation = Activation(sum + perceptron.Bias!.Value);
            }

            currentLayer = currentLayer.NextLayer;
        }
    }

    private void PropagateBackward(double[] expected)
    {
        if (expected.Length != OutputLayer.Perceptrons.Length)
        {
            throw new Exception("Invalid expected array size.");
        }

        var initialGradientMultipliers = Enumerable.Range(0, OutputLayer.Perceptrons.Length)
            .Select(i => LossDerivative(OutputLayer.Perceptrons[i].Activation, expected[i], OutputLayer.Perceptrons.Length))
            .ToArray();

        GradientDescend(OutputLayer, initialGradientMultipliers);
    }

    private void GradientDescend(Layer layer, double[] gradientMultipliers)
    {
        // All gradients are calculated, adjust parameters
        if (layer.PrevousLayer is null)
        {
            AdjustParameters();
            return;
        }

        var nextGradientMultipliers = new double[layer.PrevousLayer.Perceptrons.Length];
        for (int j = 0; j < layer.Perceptrons.Length; j++)
        {
            var perceptron = layer.Perceptrons[j];
            var z = DotProduct(perceptron.WeightsVector!, layer.PrevousLayer.Perceptrons) + perceptron.Bias!.Value;
            var dCdG = gradientMultipliers[j] * ActivationDerivative(z);
            // optimize each weight
            for (int i = 0; i < perceptron.WeightsVector!.Length; i++)
            {
                // dz(current)/dg(prev layer) = current weight
                nextGradientMultipliers[i] += dCdG * perceptron.WeightsVector[i].Value;
                // dC/dw = dC/dg * dg/dz * dz/dw
                perceptron.WeightsVector[i].TempGradient += dCdG * layer.PrevousLayer.Perceptrons[i].Activation;
            }

            // optimize bias
            perceptron.Bias.TempGradient += dCdG;
        }

        GradientDescend(layer.PrevousLayer, nextGradientMultipliers);
    }

    private void AdjustParameters()
    {
        var layer = InputLayer.NextLayer;
        while (layer is not null)
        {
            foreach (var perceptron in layer.Perceptrons)
            {
                var delBias = perceptron.Bias!.TempGradient * LearingRate;
                perceptron.Bias.Value -= delBias;
                perceptron.Bias.TempGradient = 0;

                foreach (var weight in perceptron.WeightsVector!)
                {
                    var delWeight = weight.TempGradient * LearingRate;
                    weight.Value -= delWeight;
                    weight.TempGradient = 0;
                }
            }

            layer = layer.NextLayer;
        }
    }

    private static double DotProduct(Parameter[] weights, Perceptron[] perceptrons)
    {
        double sum = 0;
        for (int i = 0; i < weights!.Length; i++)
        {
            sum += weights[i].Value * perceptrons[i].Activation;
        }

        return sum;
    }
}