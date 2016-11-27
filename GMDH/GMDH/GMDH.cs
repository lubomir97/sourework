using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GMDH
{
    class GMDH
    {
        Random rand = new Random();
        private int inputsNum;
        private int outputsNum;
        private double learningRate = 0.96;
        private double minAlpha = 0.01;
        private double A = 0.023;
        private double alpha = 0.6;
        private double[] d;
        private double[][] w;

        public GMDH(int inputsNum, int outputsNum)
        {
            this.inputsNum = inputsNum;
            this.outputsNum = outputsNum;
            w = new double[inputsNum][];
            for (int i = 0; i < inputsNum; i++)
            {
                w[i] = new double[outputsNum];
                for (int j = 0; j < outputsNum; j++)
                {
                    w[i][j] = rand.NextDouble();
                }
            }
        }

        public void ComputeOutput(int[] vectorArray)
        {
            d = new double[inputsNum];
            for (int i = 0; i < inputsNum; i++)
            {
                for (int j = 0; j < outputsNum; j++)
                {
                    d[i] += Math.Pow((w[i][j] - vectorArray[j]), 2);
                }
            }
        }

        public void Train(int[][] input)
        {
            int dMin = 0;
            while (alpha > minAlpha)
            {
                for (int vecNum = 0; vecNum < inputsNum; vecNum++)
                {
                    ComputeOutput(input[vecNum]);
                    dMin = FindMin(d);
                    UpdateWeights(input[vecNum], dMin);
                }
                alpha = learningRate * alpha;
            }
        }

        private void UpdateWeights(int[] inputs, int dMin)
        {
            for (int i = 0; i < outputsNum; i++)
            {
                w[dMin][i] = w[dMin][i] + (alpha * (inputs[i] - w[dMin][i]));
                if (alpha > A)
                {
                    if (dMin == 0)
                    {
                        w[dMin + 1][i] = w[dMin + 1][i] + (alpha * (inputs[i] - w[dMin + 1][i]));
                    }
                    else if (dMin == inputsNum - 1)
                    {
                        w[dMin - 1][i] = w[dMin - 1][i] + (alpha * (inputs[i] - w[dMin - 1][i]));
                    }
                    else
                    {
                        w[dMin - 1][i] = w[dMin - 1][i] + (alpha * (inputs[i] - w[dMin - 1][i]));
                        w[dMin + 1][i] = w[dMin + 1][i] + (alpha * (inputs[i] - w[dMin + 1][i]));
                    }
                }
            }
        }

        private int FindMin(double[] array)
        {
            int minIndex = 0;
            for (int i = 0; i < array.Length; i++)
            {
                if (array[i] < array[minIndex])
                    minIndex = i;
            }
            return minIndex;
        }

        public void Print(int[][] input)
        {
            int dMin = 0;
            Console.WriteLine("Testing:\n");
            for (int vecNum = 0; vecNum < inputsNum; vecNum++)
            {
                ComputeOutput(input[vecNum]);
                dMin = FindMin(d);
                for (int i = 0; i < inputsNum; i++)
                {
                    dMin = FindMin(d);
                }

            }
            for (int i = 0; i < inputsNum; i++)
            {
                Console.WriteLine("iteration#" + (i + 1));
                for (int j = 0; j < outputsNum; j++)
                {
                    Console.Write(w[i][j].ToString("F4") + ", ");
                }
                Console.WriteLine();
            }

            Console.WriteLine(Environment.NewLine + "Testing Result:");
            for (int vecNum = 0; vecNum < (input[0].Length - 1); vecNum++)
            {
                ComputeOutput(input[vecNum]);
                dMin = FindMin(d);

                Console.WriteLine("Expected coef А(" + vecNum + "): " + dMin);
            }
        }
    }
}
