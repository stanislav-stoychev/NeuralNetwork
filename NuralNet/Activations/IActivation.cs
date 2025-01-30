namespace NN;

public interface IActivation
{
    double Activation(double input);

    double ActivationDerivative(double input);
}