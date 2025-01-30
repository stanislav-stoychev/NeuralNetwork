using System.Reflection.Metadata;

namespace NN;

public class Perceptron
{
    public double Activation { get; set; }

    public Parameter[]? WeightsVector { get; set; }

    public Parameter? Bias { get; set; }
}