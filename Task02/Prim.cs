using System;
using System.Threading;
using System.IO;

namespace Task02
{
    public static class Prim
    {
        private static int ParallelPrim(this Graph graph)
        {
            int weight = 0;
            int[] minDist = new int[graph.VerticesCount];
            int[] minEdgeFrom = new int[graph.VerticesCount];
            bool[] used = new bool[graph.VerticesCount];

            for (int i = 0; i < graph.VerticesCount; i++)
            {
                minDist[i] = Int32.MaxValue;
                used[i] = false;
            }

            minDist[0] = 0;
            minEdgeFrom[0] = -1;
            
            for (var i = 0; i < graph.VerticesCount; i++)
            {
                var u = -1;

                for (int j = 0; j < graph.VerticesCount; j++)
                {
                    if (!used[j] && (u == -1 || minDist[j] < minDist[u]))
                    {
                        u = j;
                    }
                }
                
                used[u] = true;
                weight += minDist[u];
                var isDone = 0;
                ManualResetEvent allCompleted = new ManualResetEvent(initialState: false);

                for (var v = 0; v < graph.VerticesCount; v++)
                {
                    var v1 = v;
                    ThreadPool.QueueUserWorkItem(_ =>
                    {
                        if (graph.AdjacencyMatrix[v1, u] < minDist[v1] && !used[v1])
                        {
                            minDist[v1] = graph.AdjacencyMatrix[v1, u];
                            minEdgeFrom[u] = v1;
                        }
                        if (Interlocked.Increment(ref isDone) == graph.VerticesCount)
                        {
                            allCompleted.Set();
                        }
                    });
                }
                allCompleted.WaitOne();
            }
            return weight;
        }
        
        public static void PrintPrim(this Graph graph)
        {
            using var writer = new StreamWriter("Prim.txt");
            int result = graph.ParallelPrim();
            writer.Write($"{result} ");
        }
    }
}