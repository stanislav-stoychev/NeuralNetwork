namespace NN;

public interface ILoss
{
    double Loss(double predicted, double expected);

    double LossDerivative(double predictedDer, double expectedDer, int numberOfOutputs);
}