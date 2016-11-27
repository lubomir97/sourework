using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GMDH
{
    class Program
    {
        static void Main(string[] args)
        {
            int[][] input = new int[][] {
            new int[] {0, 1, 1, 1, 0, 0, 0},
            new int[] {0, 0, 1, 0, 1, 0, 1},
            new int[] {1, 0, 1, 0, 1, 0, 0},
            new int[] {0, 0, 0, 1, 0, 1, 1},
            new int[] {1, 1, 0, 0, 1, 0, 0},
            new int[] {0, 1, 1, 0, 1, 1, 0},
            new int[] {1, 0, 0, 1, 0, 0, 1}
         };

            GMDH gmdh = new GMDH(input.Length, input[0].Length);
            gmdh.Train(input);
            gmdh.Print(input);
            Console.ReadKey();
        }
    }
}
