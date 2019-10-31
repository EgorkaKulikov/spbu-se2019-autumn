using System;
using System.Threading;
using System.IO;

namespace Floyd
{
    public class Graph
    {
        private int N = 0;
        private int[,] adjMatrix;
        
        public void AdjacencyMatrix()
        {
            if (Console.IsInputRedirected)
            {
                N = Convert.ToInt32(Console.ReadLine());
            }
            adjMatrix = new int[N, N];
            string s;
            string[] edges = new string[3];
            int i, j, w;
            for (i = 0; i < N; i++)
            {
                for (j = 0; j < N; j++)
                {
                    if (i != j) adjMatrix[i, j] = -1;
                    else adjMatrix[i, j] = 0;
                }
            }
            if (Console.IsInputRedirected)
            {
                s = Console.ReadLine();
                while (s != null)
                {
                    edges = s.Split(' ');
                    i = Convert.ToInt32(edges[0]);
                    j = Convert.ToInt32(edges[1]);
                    w = Convert.ToInt32(edges[2]);
                    adjMatrix[i, j] = w;
                    adjMatrix[j, i] = w;
                    s = Console.ReadLine();
                }
            }
        }
        public void Floyd()
        {
            for (int k = 0; k < N; k++)
            {
                Thread th = new Thread(new ParameterizedThreadStart(OneThreadFloyd));
                th.Start(k);
                th.Join();
            }
            using (StreamWriter file = new StreamWriter("output"))
            {
                for (int i = 0; i < N; i++)
                {
                    for (int j = 0; j < N; j++)
                    {
                        file.Write(adjMatrix[i,j] + " ");
                    }
                    file.WriteLine();
                }
            }
        }
        private void OneThreadFloyd(Object ver)
        {
            int k = (int)ver;
            for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                if (adjMatrix[i, j] > 0 && adjMatrix[i, k] > 0 && adjMatrix[k, j] > 0)
                {
                    adjMatrix[i, j] = (adjMatrix[i, j] <= adjMatrix[i, k] + adjMatrix[k, j])
                        ? adjMatrix[i, j]
                        : adjMatrix[i, k] + adjMatrix[k, j];
                }
        }
    }

    internal class Program
    {
        public static void Main(string[] args)
        {
            Graph graph = new Graph();
            graph.AdjacencyMatrix();
            graph.Floyd();
        }
    }
}