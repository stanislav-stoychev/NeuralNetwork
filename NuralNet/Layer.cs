namespace NN;

public class Layer(int size)
{
    public Layer? PreviousLayer { get; set; }

    public Layer? NextLayer { get; set; }

    public Perceptron[] Perceptrons { get; set; } = new Perceptron[size];
}