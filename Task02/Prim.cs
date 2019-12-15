using System.Threading;
using System;

namespace Task02
{
    class Prim
    {
        public static int Run(Graph graph)
        {
            int size   = graph.VerticesNum;
            var matrix = graph.GetMatrix();
            int INF    = Int32.MaxValue;

            for (int i = 0; i < size; i++)
                for (int j = 0; j < size; j++)
                {
                    if (matrix[i, j] == 0 && i != j)
                        matrix[i, j] = INF;
                }
            
            int result = 0;
            bool[] added     = new bool[matrix.Length];
            int[]  minWeight = new int[matrix.Length];
            int[]  minEdgeTo = new int[matrix.Length];

            for (var i = 0; i < size; i++)
            {
                added[i] = false;
                minWeight[i] = INF;
                minEdgeTo[i] = -1;
            }

            minWeight[0] = 0;

            for (var i = 0; i < size; i++)
            {
                var v = -1;

                for (int j = 0; j < size; j++)
                {
                    if (!added[j] && (v == -1 || minWeight[j] < minWeight[v]))
                    {
                        v = j;
                    }
                }

                if (minWeight[i] == INF)
                    return INF;

                added[v] = true;
                result += minWeight[v];
                var completed = 0;
                ManualResetEvent allCompleted = new ManualResetEvent(initialState: false);

                for (var to = 0; to < size; to++)
                {
                    ThreadPool.QueueUserWorkItem(objectTo =>
                    {
                        var actualTo = (int)objectTo;
                        if (matrix[v, actualTo] < minWeight[actualTo])
                        {
                            minWeight[actualTo] = matrix[v, actualTo];
                            minEdgeTo[actualTo] = v;
                        }
                        if (Interlocked.Increment(ref completed) == size)
                        {
                            allCompleted.Set();
                        }
                    }, to);
                }

                allCompleted.WaitOne();
            }

            return result;
        }
    }
}
