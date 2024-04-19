using MathNet.Numerics.Distributions;

namespace NN;

public class NetworkBuilder
{
    private static readonly Normal _rg = new();

    private double _learningRate;

    private int _inputSize;

    private int _outputSize;

    private readonly List<int> _hiddenLayerSizes = [];

    public NetworkBuilder WithLearingRate(double learningRate)
    {
        _learningRate = learningRate;
        return this;
    }

    public NetworkBuilder WithInputSize(int size)
    {
        _inputSize = size;
        return this;
    }

    public NetworkBuilder WithOutputSize(int size)
    {
        _outputSize = size;
        return this;
    }

    public NetworkBuilder WithHiddenLayer(int size)
    {
        _hiddenLayerSizes.Add(size);
        return this;
    }

    private void Validate()
    {
        if (_learningRate <= 0)
        {
            throw new Exception("Learning rate must be greater than zero.");
        }
    }

    public NeuralNetwork BuildAndInitialize()
    {
        Validate();

        var network = new NeuralNetwork(_inputSize)
        {
            LearingRate = _learningRate,
            OutputLayerSize = _outputSize
        };

        var currentLayer = network.InputLayer;

        // Init input layer perceptrons
        currentLayer.Perceptrons = Enumerable.Repeat(0, currentLayer.Perceptrons.Length)
            .Select(i => new Perceptron())
            .ToArray();

        // Init hidden layers + output layer
        foreach (var layerSize in _hiddenLayerSizes.Concat([_outputSize]))
        {
            currentLayer.NextLayer = new(layerSize)
            {
                PrevousLayer = currentLayer
            };

            currentLayer.NextLayer.Perceptrons = Enumerable.Repeat(0, currentLayer.NextLayer.Perceptrons.Length)
                .Select(i => new Perceptron()
                {
                    // Init random weights around the normal distribution for each perceptron and bias
                    WeightsVector = Enumerable.Repeat(0, currentLayer.Perceptrons.Length)
                        .Select(i => new Parameter { Value = _rg.Sample() })
                        .ToArray(),
                    Bias = new() { Value = 0 }
                })
                .ToArray();

            currentLayer = currentLayer.NextLayer;
        }

        return network;
    }
}