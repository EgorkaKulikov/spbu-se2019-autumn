using System;
using System.IO;

namespace Task02
{
    public class GenerateMatrix
    {
        static int Whole = 1000000;

        public static void Execute(int n, int m, int minWeight, int maxWeight)
        {
            GeneralResources.n = n;
            GeneralResources.graphMatrix = new int[n, n];
            Random random = new Random();
            int partAmountEdgesFromMax = (int)((long)m * Whole / (n * (n - 1) / 2));
            m = 0;
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    if (random.Next(Whole) < partAmountEdgesFromMax)
                    {
                        int weight = random.Next(minWeight, maxWeight);
                        GeneralResources.graphMatrix[i, j] = GeneralResources.graphMatrix[j, i] = weight;
                        m++;
                    }
                    else
                    {
                        GeneralResources.graphMatrix[i, j] = GeneralResources.graphMatrix[j, i] = GeneralResources.INF;
                    }
                }
            }

            GeneralResources.graphListEdges = new (int, int, int)[m];
            int k = 0;
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    if (GeneralResources.graphMatrix[i, j] != GeneralResources.INF)
                    {
                        GeneralResources.graphListEdges[k] = (GeneralResources.graphMatrix[i, j], i, j);
                        k++;
                    }
                }
            }

            StreamWriter foutMatrix = new StreamWriter("matrix.txt");
            foutMatrix.Write($"{n} {m}\n");
            for (k = 0; k < m; k++)
            {
                foutMatrix.Write($"{GeneralResources.graphListEdges[k].Item2 + 1} " +
                    $"{GeneralResources.graphListEdges[k].Item3 + 1} " +
                    $"{GeneralResources.graphListEdges[k].Item1}\n");
            }
            foutMatrix.Close();
        }
    }
}
