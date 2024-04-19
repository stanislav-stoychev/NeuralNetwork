using NN;

//var network = new NetworkBuilder()
//   .WithInputSize(784)
//   .WithHiddenLayer(8)
//   .WithOutputSize(10)
//   .WithLearingRate(0.02)
//   .BuildAndInitialize();
//
//var trainingData = NetworkLoader.LoadVectorsFromFile(@"Data\mnist_train.csv", 10);
//network.Train(trainingData, 1000);
var network = await NetworkLoader.LoadFromFile(@"Data\trained.ml");
var testData = NetworkLoader.LoadVectorsFromFile(@"Data\mnist_test.csv", 10);
network.Test(testData);

await NetworkLoader.LoadToFile(network, @"Data\trained.ml");