using System;
using System.Threading;
using System.IO;

namespace Floyd
{
    public class Graph
    {
        private int vertexCount = 0;
        private int[,] adjMatrix;
        
        public Graph()
        {
            if (Console.IsInputRedirected)
            {
                vertexCount = Convert.ToInt32(Console.ReadLine());
            }
            adjMatrix = new int[vertexCount, vertexCount];
            string s;
            string[] edges = new string[3];
            int i, j, w;
            for (i = 0; i < vertexCount; i++)
            {
                for (j = 0; j < vertexCount; j++)
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
            int completed = 0;
            ManualResetEvent allDone = new ManualResetEvent(initialState: false);
            for (int k = 0; k < vertexCount; k++)
            {
                Thread th = new Thread( vertexBetween =>
                {
                    FindTheShortestPaths(vertexBetween);
                    if (Interlocked.Increment(ref completed) == vertexCount)
                    {
                        allDone.Set();
                    }
                });
                th.Start(k);
            }
            allDone.WaitOne();
        }
        
        private void FindTheShortestPaths(object vertexBetween)
        {
            int k = (int)vertexBetween;
            for (int i = 0; i < vertexCount; i++)
            for (int j = 0; j < vertexCount; j++)
                if (adjMatrix[i, j] > 0 && adjMatrix[i, k] > 0 && adjMatrix[k, j] > 0)
                {
                    adjMatrix[i, j] = (adjMatrix[i, j] <= adjMatrix[i, k] + adjMatrix[k, j])
                        ? adjMatrix[i, j]
                        : adjMatrix[i, k] + adjMatrix[k, j];
                }
        }

        public void Output()
        {
            using (StreamWriter file = new StreamWriter("output"))
            {
                for (int i = 0; i < vertexCount; i++)
                {
                    for (int j = 0; j < vertexCount; j++)
                    {
                        file.Write(adjMatrix[i,j] + " ");
                    }
                    file.WriteLine();
                }
            }
        }
    }

    internal class Program
    {
        public static void Main(string[] args)
        {
            Graph graph = new Graph();
            graph.Floyd();
            graph.Output();
        }
    }
}