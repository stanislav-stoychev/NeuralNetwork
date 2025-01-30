using NN;

namespace NeuralNet.Activations;

public class ReLUActivation : IActivation
{
    public double Activation(double input) => input >= 0 ? input : 0;

    public double ActivationDerivative(double input) => input >= 0 ? 1 : 0;
}
