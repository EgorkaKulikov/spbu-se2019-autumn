using System;
using System.IO;
using System.Threading;

namespace Task02
{
    class Prim
    {
        public int primAlg(int[,] graph, int size)
        {
            bool[] used = new bool[size];
            int[] minDist = new int[size];
            for (int i = 0; i < size; i++)
            {
                used[i] = false;
                minDist[i] = Int32.MaxValue;
            }
            minDist[0] = 0;
            int weight = 0;
            for (int i = 0; i < size; i++)
            {
                int curr = -1;
                for (int j = 0; j < size; j++)
                    if (!used[j] && (curr == -1 || minDist[j] < minDist[curr]))
                        curr = j;
                weight += minDist[curr];
                used[curr] = true;
                AutoResetEvent done = new AutoResetEvent(false);
                int numOfDone = 0;
                for (int j = 0; j < size; j++)
                {
                    int tmp = j;
                    ThreadPool.QueueUserWorkItem(_ =>
                    {
                        if (graph[curr, tmp] < minDist[tmp])
                            minDist[tmp] = graph[curr, tmp];
                        if (Interlocked.Increment(ref numOfDone) == size)
                            done.Set();
                    });
                }
                done.WaitOne();
            }
            return weight;
        }

        public void printResult(int[,] graph, int size)
        {
            using (var sw = new StreamWriter("Prim.txt"))
            {
                var result = primAlg(graph, size);
                sw.Write(result);
            }
        }
    }
}
