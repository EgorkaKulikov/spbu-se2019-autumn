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

        public Graph(int amountVertexes, int approximateAmountEdges, int minWeight, int maxWeight)
        {
            graphAmountVertexes = amountVertexes;
            graphMatrix = new int[amountVertexes, amountVertexes];
            Random random = new Random();
            int partAmountEdgesFromMax = (int)((long)approximateAmountEdges * NumberOfConventionalUnitsInTotal /
                (amountVertexes * (amountVertexes - 1) / 2));
            int exectAmountEdges = 0;
            for (int i = 0; i < amountVertexes; i++)
            {
                for (int j = i + 1; j < amountVertexes; j++)
                {
                    if (random.Next(NumberOfConventionalUnitsInTotal) < partAmountEdgesFromMax)
                    {
                        int weight = random.Next(minWeight, maxWeight);
                        graphMatrix[i, j] = graphMatrix[j, i] = weight;
                        exectAmountEdges++;
                    }
                    else
                    {
                        graphMatrix[i, j] = graphMatrix[j, i] = INF;
                    }
                }
            }

            graphListEdges = new (int, int, int)[exectAmountEdges];
            int k = 0;
            for (int i = 0; i < amountVertexes; i++)
            {
                for (int j = i + 1; j < amountVertexes; j++)
                {
                    if (graphMatrix[i, j] != INF)
                    {
                        graphListEdges[k] = (graphMatrix[i, j], i, j);
                        k++;
                    }
                }
            }

            StreamWriter foutMatrix = new StreamWriter("matrix.txt");
            foutMatrix.Write($"{amountVertexes} {exectAmountEdges}\n");
            for (k = 0; k < exectAmountEdges; k++)
            {
                foutMatrix.Write($"{graphListEdges[k].Item2 + 1} " +
                    $"{graphListEdges[k].Item3 + 1} " +
                    $"{graphListEdges[k].Item1}\n");
            }
            foutMatrix.Close();
        }
    }
}
