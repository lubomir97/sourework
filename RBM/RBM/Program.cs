using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RBM
{
    class Program
    {
        static void Main(string[] args)
        {
            Random rand = new Random();
            double[][] data = new double[20][];
            double[][] expectedAnswers = new double[20][];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = new double[3];
                for (int j = 0; j < data[i].Length; j++)
                {
                    data[i][j] = rand.NextDouble();
                }
                switch (i + 1)
                {
                    case 1:
                        while (data[i][2] > data[i][1] || data[i][2] > data[i][0])
                            data[i][2] /= 2;
                        break;
                    case 3:
                        data[i][2] = data[i][1] + data[i][0];
                        break;
                    case 5:
                        data[i][0] = Math.Abs(data[i][1] - data[i][2]);
                        break;
                    case 7:
                        data[i][1] = data[i][0] - data[i][2];
                        break;
                    case 9:
                        data[i][0] = 3 * rand.Next();
                        data[i][1] = 3 * rand.Next();
                        data[i][2] = 3 * rand.Next();
                        break;
                }
            }
            expectedAnswers = CalculateExpected(data);
            RestrictedBoltzmannMachine machine = new RestrictedBoltzmannMachine(3, 3, 3);
            machine.Train(data, expectedAnswers, 2.5);

            double[][] controlData = new double[10][];
            for (int i = 0; i < controlData.Length; i++)
            {
                controlData[i] = new double[3];
                for (int j = 0; j < controlData[i].Length; j++)
                {
                    controlData[i][j] = rand.NextDouble();
                }
            }
            expectedAnswers = CalculateExpected(controlData);
            for (int i = 0; i < controlData.Length; i++)
            {
                double[] computed = machine.ComputeOutputs(controlData[i]);
                Console.Write("Computed:");
                PrintVector(computed);
                Console.Write("  Expected:");
                PrintVector(expectedAnswers[i]);
                Console.WriteLine();
            }
        }

        static double[][] CalculateExpected(double[][] data)
        {
            Random rand = new Random();
            double[][] expected = new double[data.Length][];
            double mean = 0, sum = 0;
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = new double[3];
                for (int j = 0; j < data[i].Length; j++)
                {
                    data[i][j] = rand.NextDouble();
                }
                sum += data[i][0];
            }
            mean = sum / data.Length;
            for (int i = 0; i < data.Length; i++)
            {
                expected[i] = new double[3];
                expected[i] = new double[] {
               (Math.Sin(data[i][0]) + Math.Cos(data[i][1]) + Math.Sin(data[0][2])) / 3,
               data[i][0] > mean ? 0.2 : 0.4,
               Math.Round(data[i][0],1) % 0.2 == 0 ? 0.2 : 0.4
            };
            }
            return expected;
        }
        static void PrintVector(double[] vector)
        {
            foreach (var item in vector)
            {
                Console.Write(item.ToString("0.00000") + " ");
            }
        }
    }
}
