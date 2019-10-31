using System;
using System.Threading;
using System.IO;

namespace Prim
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

        public void Prim()
        {
            int weightMST = 0;
            int i, completed = 0;
            ManualResetEvent allDone = new ManualResetEvent(initialState: false);
            int[] min = new int[100];
            Mutex mutex = new Mutex();
            for (i = 0; i < N - 1; i++)
            {
                ThreadPool.QueueUserWorkItem(_ =>
                {
                    min[Thread.CurrentThread.ManagedThreadId] = 0;
                    for (int j = 0; j < N; j++)
                    {
                        if (adjMatrix[i, j] > 0
                            && (adjMatrix[i, j] < min[Thread.CurrentThread.ManagedThreadId]
                                || min[Thread.CurrentThread.ManagedThreadId] == 0))
                        {
                            min[Thread.CurrentThread.ManagedThreadId] = adjMatrix[i, j];
                            if (i < j) adjMatrix[j, i] = adjMatrix[j, i] * (-1);
                        }
                    }

                    mutex.WaitOne();
                    weightMST += min[Thread.CurrentThread.ManagedThreadId];
                    mutex.ReleaseMutex();

                    if (Interlocked.Increment(ref completed) == N - 1)
                    {
                        allDone.Set();
                    }
                });
            }

            allDone.WaitOne();
            using (StreamWriter file = new StreamWriter("output"))
            {
                file.Write(weightMST);
            }
        }
    }

    internal class Program
    {
        public static void Main(string[] args)
        {
            Graph graph = new Graph();
            graph.AdjacencyMatrix();
            graph.Prim();
        }
    }
}