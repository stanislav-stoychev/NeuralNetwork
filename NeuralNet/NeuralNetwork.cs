﻿namespace NN;

public class NeuralNetwork(IActivation activation, ILoss loss, int inputLayerSize)
{
    private readonly IActivation _activation = activation;
    private readonly ILoss _loss = loss;

    public double LearningRate { get; set; } = 0.01;

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
                .Select(i => _loss.Loss(OutputLayer.Perceptrons[i].Activation, data.result[i]))
                .ToArray()
                .Sum() / OutputLayer.Perceptrons.Length;

            Console.WriteLine($"Total cost: {totalCost} epoch: {i + 1}.");
        }
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
                double sum = DotProduct(perceptron.WeightsVector!, currentLayer.PreviousLayer!.Perceptrons);
                perceptron.Activation = _activation.Activation(sum + perceptron.Bias!.Value);
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
            .Select(i => _loss.LossDerivative(OutputLayer.Perceptrons[i].Activation, expected[i], OutputLayer.Perceptrons.Length))
            .ToArray();

        GradientDescend(OutputLayer, initialGradientMultipliers);
    }

    private void GradientDescend(Layer layer, double[] gradientMultipliers)
    {
        // All gradients are calculated, adjust parameters
        if (layer.PreviousLayer is null)
        {
            AdjustParameters();
            return;
        }

        var nextGradientMultipliers = new double[layer.PreviousLayer.Perceptrons.Length];
        for (int j = 0; j < layer.Perceptrons.Length; j++)
        {
            var perceptron = layer.Perceptrons[j];
            var z = DotProduct(perceptron.WeightsVector!, layer.PreviousLayer.Perceptrons) + perceptron.Bias!.Value;
            var dCdG = gradientMultipliers[j] * _activation.ActivationDerivative(z);
            // optimize each weight
            for (int i = 0; i < perceptron.WeightsVector!.Length; i++)
            {
                // dz(current)/dg(prev layer) = current weight
                nextGradientMultipliers[i] += dCdG * perceptron.WeightsVector[i].Value;
                // dC/dw = dC/dg * dg/dz * dz/dw
                perceptron.WeightsVector[i].TempGradient += dCdG * layer.PreviousLayer.Perceptrons[i].Activation;
            }

            // optimize bias
            perceptron.Bias.TempGradient += dCdG;
        }

        GradientDescend(layer.PreviousLayer, nextGradientMultipliers);
    }

    private void AdjustParameters()
    {
        var layer = InputLayer.NextLayer;
        while (layer is not null)
        {
            foreach (var perceptron in layer.Perceptrons)
            {
                var delBias = perceptron.Bias!.TempGradient * LearningRate;
                perceptron.Bias.Value -= delBias;
                perceptron.Bias.TempGradient = 0;

                foreach (var weight in perceptron.WeightsVector!)
                {
                    var delWeight = weight.TempGradient * LearningRate;
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