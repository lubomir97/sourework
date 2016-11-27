using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RBM
{
    class RestrictedBoltzmannMachine
    {
        private int numInput;  // Кількість нейронів у вхідному шарі
        private int numHidden; // Кількість нейронів у прихованому шарі
        private int numOutput; // Кількість нейронів у вихідному шарі

        private double[] inputs;        // Масив з вхідними даними
        private double[][] ihWeights;   // Ваги для переходу між вхідним та прихованим шарами
        private double[] hiddenBiases;  // Зсуви(w0) прихованого шару
        private double[] hiddenSums;    // Суми прихованого шару (S1)
        private double[] hiddenOutputs; // Виходи прихованого шару

        private double[][] hoWeights;   // Ваги для переходу між прихованим та вихідним шарами
        private double[] outputBiases;  // Зсуви(w0) вихідного шару
        private double[] outputSums;    // Суми вихідного шару (S2)
        private double[] outputs;       // Виходи

        private double[] outputGradients; // Градієнт вихідного шару
        private double[] hiddenGradients; // Градієнт прихованого шару

        private double[][] ihPrevWeightsDelta;  // попереднє значення зміни ваг(вхід-прихов)
        private double[] hiddenPrevBiasesDelta; // попереднє значення зміни зсувів (прихов)
        private double[][] hoPrevWeightsDelta;  // попереднє значення зміни ваг(прихов-вихід)
        private double[] outputPrevBiasesDelta; // попереднє значення зміни зсувів (вихід)

        private double T;

        public RestrictedBoltzmannMachine(int numInput, int numHidden, int numOutput)
        {
            this.numInput = numInput;
            this.numHidden = numHidden;
            this.numOutput = numOutput;

            inputs = new double[numInput];
            ihWeights = MakeMatrix(numInput, numHidden);
            hiddenBiases = new double[numHidden];
            hiddenSums = new double[numHidden];

            hiddenOutputs = new double[numHidden];
            hoWeights = MakeMatrix(numHidden, numOutput);
            outputBiases = new double[numOutput];
            outputSums = new double[numOutput];
            outputs = new double[numOutput];

            outputGradients = new double[numOutput];
            hiddenGradients = new double[numHidden];

            ihPrevWeightsDelta = MakeMatrix(numInput, numHidden);
            hiddenPrevBiasesDelta = new double[numHidden];
            hoPrevWeightsDelta = MakeMatrix(numHidden, numOutput);
            outputPrevBiasesDelta = new double[numOutput];

            int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
            double[] weights = new double[numWeights];
            FillArrayWithRandomNumbers(weights, 0.1, 0.09);
            SetWeights(weights);
        }

        //   Задає значення ваг
        public void SetWeights(double[] weights)
        {
            int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
            if (weights.Length != numWeights)
                throw new Exception("The weights array length: " + weights.Length +
                  " does not match the total number of weights and biases: " + numWeights);

            int k = 0;

            for (int i = 0; i < numInput; ++i)
                for (int j = 0; j < numHidden; ++j)
                    ihWeights[i][j] = weights[k++];

            for (int i = 0; i < numHidden; ++i)
                hiddenBiases[i] = weights[k++];

            for (int i = 0; i < numHidden; ++i)
                for (int j = 0; j < numOutput; ++j)
                    hoWeights[i][j] = weights[k++];

            for (int i = 0; i < numOutput; ++i)
                outputBiases[i] = weights[k++];
        }

        //   Повертає значення ваг
        public double[] GetWeights()
        {
            int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
            double[] result = new double[numWeights];
            int k = 0;
            for (int i = 0; i < ihWeights.Length; ++i)
                for (int j = 0; j < ihWeights[0].Length; ++j)
                    result[k++] = ihWeights[i][j];
            for (int i = 0; i < hiddenBiases.Length; ++i)
                result[k++] = hiddenBiases[i];
            for (int i = 0; i < hoWeights.Length; ++i)
                for (int j = 0; j < hoWeights[0].Length; ++j)
                    result[k++] = hoWeights[i][j];
            for (int i = 0; i < outputBiases.Length; ++i)
                result[k++] = outputBiases[i];
            return result;
        }

        //   Повертає результат обрахунку
        public double[] GetOutputs()
        {
            return (double[])outputs.Clone();
        }

        //   Метод для навчання нейронної мережі за допомогою заданої навчальної вибірки
        //   і заданого коефіцієнту швидкості навчання
        public int Train(double[][] input, double[][] expectedValues, double learnRate)
        {
            T = 273;
            double momentum = 0.3;
            double totalError = double.MaxValue;
            int totalIterations = 0;
            double[] output;

            //   Повторюємо цикл навчання по всій вибірці, поки 
            //   сумарна помилка не зменшиться до значення 4 або менше
            while (totalError > 1)
            {
                totalError = 0;
                //   Цикл, який по черзі передає всі значення з навчальної вибірки
                //   методу обрахунку, порівнює результат із правильним (очікуваним)
                //   і, якщо необхідно, регулює ваги.
                for (int i = 0; i < input.Length; i++)
                {
                    //   Максимальна допустима кількість ітерацій
                    int maxIterations = 10000;
                    //   Допустима величина помилки
                    double targetError = 0.001;
                    //   Номер поточної ітерації
                    int iteration = 0;
                    //   Поточне значення помилки
                    double prevError = double.MaxValue;
                    double error = double.MaxValue;

                    while (iteration < maxIterations)
                    {
                        totalIterations++;
                        output = ComputeOutputs(input[i]);
                        error = Error(expectedValues[i], output);
                        //   Якщо досягнуто допустимого рівня помилки - вихід з циклу
                        if (error < targetError) break;
                        totalError += error;
                        if (error > prevError)
                        {

                        }
                        ++iteration;
                        UpdateWeights(input[i], expectedValues[i], learnRate, momentum, error);
                    }
                }
                System.Console.WriteLine("Total Error: " + totalError);

            }
            //   Вивід даних про результати навчання
            System.Console.WriteLine("Total Error: " + totalError);
            System.Console.WriteLine("Completed training in " + totalIterations + " iterations");
            return totalIterations;
        }

        //   Обрахунок виводу
        public double[] ComputeOutputs(double[] xValues)
        {
            if (xValues.Length != numInput)
                throw new Exception("Invalid input length " + inputs.Length);

            hiddenSums = new double[numHidden];   // S1
            outputSums = new double[numOutput]; // S2
            inputs = (double[])xValues.Clone(); // y0

            for (int i = 0; i < numHidden; ++i)
            {
                for (int j = 0; j < numInput; ++j)
                {
                    hiddenSums[i] += inputs[j] * ihWeights[j][i];
                }
                hiddenSums[i] += hiddenBiases[i];
                hiddenOutputs[i] = HyperTanFunction(hiddenSums[i]);
            }

            for (int i = 0; i < numOutput; ++i)
            {
                for (int j = 0; j < numHidden; ++j)
                {
                    outputSums[i] += hiddenOutputs[j] * hoWeights[j][i];
                }
                outputSums[i] += outputBiases[i];
                outputs[i] = Sigmoid(outputSums[i]);
            }

            return (double[])outputs.Clone();
        }

        private static double Sigmoid(double x)
        {
            if (x < -45) return 0;
            else if (x > 45) return 1;
            else return 1.0 / (1 + Math.Exp(-x));
        }

        private static double HyperTanFunction(double x)
        {
            if (x < -45.0) return -1.0;
            else if (x > 45.0) return 1.0;
            else return Math.Tanh(x);
        }

        public void UpdateWeights(double[] xValues, double[] tValues, double learningRate, double mom, double error)
        {
            if (tValues.Length != numOutput)
                throw new Exception("target values not same Length as output in UpdateWeights");
            double k = 0.001;
            for (int i = 0; i < outputGradients.Length; ++i)
            {
                outputGradients[i] = (1 - outputs[i]) * outputs[i] * (tValues[i] - outputs[i]);
            }

            for (int i = 0; i < hiddenGradients.Length; ++i)
            {
                double sum = 0.0;
                for (int j = 0; j < numOutput; ++j)
                    sum += outputGradients[j] * hoWeights[i][j];
                hiddenGradients[i] = (1 - hiddenOutputs[i]) * (1 + hiddenOutputs[i]) * sum;
            }

            for (int i = 0; i < ihWeights.Length; ++i)
            {
                for (int j = 0; j < ihWeights[0].Length; ++j)
                {
                    double delta = learningRate * hiddenGradients[j] * inputs[i];
                    double prevWeight = ihWeights[i][j];
                    double prevOutput = ComputeOutputs(xValues)[0];
                    ihWeights[i][j] += delta + mom * ihPrevWeightsDelta[i][j];
                    if (error < Math.Abs(ComputeOutputs(xValues)[0] - prevOutput))
                    {
                        double sum = 0;
                        for (int l = 0; l < ihWeights[i].Length; l++)
                        {
                            sum += ihWeights[i][l];
                        }
                        double P = Math.Exp(-sum / k * T);
                        Random rand = new Random();
                        if (P < rand.NextDouble())
                        {
                            ihWeights[i][j] = prevWeight;
                            break;
                        }
                    }
                    ihPrevWeightsDelta[i][j] = delta;
                }
            }

            for (int i = 0; i < hiddenBiases.Length; ++i)
            {
                double delta = learningRate * hiddenGradients[i];
                double prevWeight = hiddenBiases[i];
                double prevOutput = ComputeOutputs(xValues)[0];
                hiddenBiases[i] += delta + mom * hiddenPrevBiasesDelta[i];
                if (error < Math.Abs(ComputeOutputs(xValues)[0] - prevOutput))
                {
                    double P = Math.Exp(-hiddenBiases[i] / k * T);
                    Random rand = new Random();
                    if (P < rand.NextDouble())
                    {
                        hiddenBiases[i] = prevWeight;
                        break;
                    }
                }
                hiddenPrevBiasesDelta[i] = delta;
            }

            for (int i = 0; i < hoWeights.Length; ++i)
            {
                for (int j = 0; j < hoWeights[0].Length; ++j)
                {
                    double delta = learningRate * outputGradients[j] * hiddenOutputs[i];
                    double prevWeight = hoWeights[i][j];
                    double prevOutput = ComputeOutputs(xValues)[0];
                    hoWeights[i][j] += delta + mom * ihPrevWeightsDelta[i][j];
                    if (error < Math.Abs(ComputeOutputs(xValues)[0] - prevOutput))
                    {
                        double sum = 0;
                        for (int l = 0; l < hoWeights[i].Length; l++)
                        {
                            sum += hoWeights[i][l];
                        }
                        double P = Math.Exp(-sum / k * T);
                        Random rand = new Random();
                        if (P < rand.NextDouble())
                        {
                            hoWeights[i][j] = prevWeight;
                            break;
                        }
                    }
                }
            }

            for (int i = 0; i < outputBiases.Length; ++i)
            {
                double delta = learningRate * outputGradients[i] * 1.0;
                double prevWeight = outputBiases[i];
                outputBiases[i] += delta + mom * outputPrevBiasesDelta[i];
                double prevOutput = ComputeOutputs(xValues)[0];
                if (error < Math.Abs(ComputeOutputs(xValues)[0] - prevOutput))
                {
                    double P = Math.Exp(-outputBiases[i] / k * T);
                    Random rand = new Random();
                    if (P < rand.NextDouble())
                    {
                        outputBiases[i] = prevWeight;
                        break;
                    }
                }
                outputPrevBiasesDelta[i] = delta;
            }
        }

        public static void FillArrayWithRandomNumbers(double[] array, double multiplier, double offset)
        {
            Random random = new Random();
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = multiplier * random.NextDouble() + offset;
            }
        }
        public static double Error(double[] tValues, double[] yValues)
        {
            double sum = 0.0;
            for (int i = 0; i < tValues.Length; ++i)
                sum += (tValues[i] - yValues[i]) * (tValues[i] - yValues[i]);
            return Math.Sqrt(sum);
        }
        public static double[][] MakeMatrix(int rows, int cols)
        {
            double[][] result = new double[rows][];
            for (int i = 0; i < rows; ++i)
                result[i] = new double[cols];
            return result;
        }
    }
}
