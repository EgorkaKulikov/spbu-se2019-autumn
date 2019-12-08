using System;
using System.IO;

namespace Task02
{
    public class Graph
    {
        public const int INF = 1000000000;
        public int[,] graphMatrix;
        public (int, int, int)[] graphListEdges;
        public int graphAmountVertexes;
        static int NumberOfConventionalUnitsInTotal = 1000000;

        public Graph(int n, int m, int minWeight, int maxWeight)
        {
            graphAmountVertexes = n;
            graphMatrix = new int[n, n];
            Random random = new Random();
            int partAmountEdgesFromMax = (int)((long)m * NumberOfConventionalUnitsInTotal / (n * (n - 1) / 2));
            m = 0;
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    if (random.Next(NumberOfConventionalUnitsInTotal) < partAmountEdgesFromMax)
                    {
                        int weight = random.Next(minWeight, maxWeight);
                        graphMatrix[i, j] = graphMatrix[j, i] = weight;
                        m++;
                    }
                    else
                    {
                        graphMatrix[i, j] = graphMatrix[j, i] = INF;
                    }
                }
            }

            graphListEdges = new (int, int, int)[m];
            int k = 0;
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    if (graphMatrix[i, j] != INF)
                    {
                        graphListEdges[k] = (graphMatrix[i, j], i, j);
                        k++;
                    }
                }
            }

            StreamWriter foutMatrix = new StreamWriter("matrix.txt");
            foutMatrix.Write($"{n} {m}\n");
            for (k = 0; k < m; k++)
            {
                foutMatrix.Write($"{graphListEdges[k].Item2 + 1} " +
                    $"{graphListEdges[k].Item3 + 1} " +
                    $"{graphListEdges[k].Item1}\n");
            }
            foutMatrix.Close();
        }
    }
}
