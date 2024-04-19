using Newtonsoft.Json;

namespace NN;

public static class NetworkLoader
{
    private const string LayerDeimeter = "NewLayer";

    public static List<(double[] input, double[] result)> LoadVectorsFromFile(string filePath, int resultSize, char inputDelimiter = ',')
    {
        var data = new List<(double[], double[])>();
        foreach (var line in File.ReadLines(filePath))
        {
            var values = line.Split(inputDelimiter);
            var result = Enumerable.Range(0, resultSize)
                .Select(i => (double)0)
                .ToArray();

            result[int.Parse(values[0])] = 1;
            data.Add((values.Skip(1).Select(double.Parse).ToArray(), result));
        }

        return data;
    }

    public static async Task LoadToFile(NeuralNetwork network, string fullFileName)
    {
        using var fileStream = File.Create(fullFileName);
        using var writer = new StreamWriter(fileStream);
        var currentLayer = network.InputLayer;
        while (currentLayer is not null)
        {
            var json = JsonConvert.SerializeObject(currentLayer.Perceptrons);
            await writer.WriteAsync(json);
            await writer.WriteAsync(LayerDeimeter);
            currentLayer = currentLayer.NextLayer;
        }
    }

    public static async Task<NeuralNetwork> LoadFromFile(string fullFileName)
    {
        var content = await File.ReadAllTextAsync(fullFileName);
        var layers = content.Split(LayerDeimeter)
            .Where(s => string.IsNullOrWhiteSpace(s) == false)
            .ToArray();

        var perceptrons = JsonConvert.DeserializeObject<Perceptron[]>(layers[0]);
        var network = new NeuralNetwork(perceptrons!.Length);

        var currentLayer = network.InputLayer;
        currentLayer.Perceptrons = perceptrons;

        foreach (var layer in layers.Skip(1))
        {
            var perc = JsonConvert.DeserializeObject<Perceptron[]>(layer)!;
            currentLayer.NextLayer = new(perc.Length)
            {
                PrevousLayer = currentLayer,
                Perceptrons = perc
            };

            currentLayer = currentLayer.NextLayer;
        }

        return network;
    }
}