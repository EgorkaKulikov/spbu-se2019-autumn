using System;

namespace Task02
{
    public class Utilities
    {
        static void PrintMSTs(int verticesNumber, int[] parent1, int[] parent2, int[,] graph)
        {
            Console.WriteLine("Edge \tWeight \tEdge \tWeight");
            for (int i = 1; i < verticesNumber; i++)
            {
                Console.Write(parent1[i] + " - " + i + "\t" + graph[parent1[i], i] + "\t");
                Console.WriteLine(parent2[i] + " - " + i + "\t" + graph[parent2[i], i]);
            }
        }

        static int CalculateMatrixMSTSum(int[] prev, int verticesNumber, int[,] graphMatrix)
        {
            int sum = 0;
            for (int i = 1; i < verticesNumber; i++)
            {
                sum += graphMatrix[prev[i], i];
            }

            return sum;
        }
    }
}