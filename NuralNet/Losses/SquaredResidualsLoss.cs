namespace NN;

public class SquaredResidualsLoss : ILoss
{    
    public double Loss(double predicted, double expected)
    {
        return Math.Pow(predicted - expected, 2);
    }

    public double LossDerivative(double predictedDer, double expectedDer, int numberOfOutputs)
    {
        return 2 / (double)numberOfOutputs * (predictedDer - expectedDer);
    }
}