namespace NN;

public class SigmoidActivation : IActivation
{
    public double Activation(double x)
    {
        return 1 / (1 + Math.Exp(-x));
    }

    public double ActivationDerivative(double x)
    {
        var activation = Activation(x);
        return activation - Math.Pow(activation, 2);
    }
}