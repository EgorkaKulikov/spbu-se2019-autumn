using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace Task02
{
    public static class Prim
    {
        public static int ExecPrim(List<Edge> edges, int verticesNum)
        {
            int[,] dist = Helper.GetIncidenceMatrix(edges, verticesNum);

            bool[] used = Enumerable.Repeat(false, verticesNum).ToArray();
            int[] minEdges = Enumerable.Repeat(It.Inf, verticesNum).ToArray();
            int[] selectedEdge = Enumerable.Repeat(-1, verticesNum).ToArray();

            int cost = 0;
            minEdges[0] = 0;

            int completed;
            const int blockSize = 100;
            ManualResetEvent allDone = new ManualResetEvent(initialState: false);
            //amount of iterations for ThreadPool
            int blocks = verticesNum / blockSize;

            for (int i = 0; i < verticesNum; i++)
            {
                completed = 0;
                allDone.Reset();

                int[] currentMins = Enumerable.Repeat(-1, blocks).ToArray();

                for (int blockIndex = 0; blockIndex < blocks; blockIndex++)
                {
                    int start = blockIndex * blockSize;
                    int end = (verticesNum - start < 2 * blockSize) ? 
                        verticesNum : start + blockSize;
                    
                    var index = blockIndex;
                    ThreadPool.QueueUserWorkItem(_ =>
                    {
                        for (int vertex = start; vertex < end; vertex++)
                        {
                            if (!used[vertex] && (currentMins[index] == -1 || minEdges[vertex] < minEdges[currentMins[index]]))
                                currentMins[index] = vertex;
                        }
                        
                        if (Interlocked.Increment(ref completed) == blocks)
                        {
                            allDone.Set();
                        }
                    }, blockIndex);
                }
                allDone.WaitOne();

                //choosing a minimal edge among selected above
                int current = -1;

                foreach (int min in currentMins)
                {
                    if (min != -1)
                        if (!used[min] && (current == -1 || minEdges[min] < minEdges[current]))
                            current = min;
                }
                
                if (current == -1)
                    continue;
                
                if (minEdges[current] != It.Inf)
                {
                    cost += minEdges[current];
                }
                used[current] = true;

                completed = 0;
                allDone.Reset();
                for (int blockIndex = 0; blockIndex < blocks; blockIndex++)
                {
                    int start = blockIndex * blockSize;
                    int end = (verticesNum - start < 2 * blockSize) ? 
                        verticesNum : start + blockSize;
                    ThreadPool.QueueUserWorkItem(_ =>
                    {
                        for (int target = start; target < end; target++)
                        {
                            if (dist[current, target] >= minEdges[target]) continue;
                            minEdges[target] = dist[current, target];
                            selectedEdge[target] = current;
                        }
                        if (Interlocked.Increment(ref completed) == blocks)
                        {
                            allDone.Set();
                        }
                    });
                }
                
                allDone.WaitOne();
                
                /*for (int target = 0; target < verticesNum; target++)
                {
                    if (dist[current, target] >= minEdges[target]) continue;
                    minEdges[target] = dist[current, target];
                    selectedEdge[target] = current;
                }*/
            }

            return cost;
        }
    }
}