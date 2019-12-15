using System;
using System.Threading;

namespace Task02
{
    public class Prim
    {
        static int MinKey(int verticesNumber, int[] key, bool[] mstSet)
        {
            int min = Int32.MaxValue, minIndex = -1;

            for (int v = 0; v < verticesNumber; v++)
                if (mstSet[v] == false && key[v] < min)
                {
                    min = key[v];
                    minIndex = v;
                }
            return minIndex;
        }

        public static int[] RunSeqPrim(int[,] graphMatrix)
        {
            int verticesNumber = graphMatrix.GetLength(0);
            int[] prev = new int[verticesNumber];
            int[] key = new int[verticesNumber];
            bool[] mstSet = new bool[verticesNumber];

            for (int i = 0; i < verticesNumber; i++)
            {
                key[i] = int.MaxValue;
                mstSet[i] = false;
            }

            key[0] = 0;
            prev[0] = -1;
            for (int count = 0; count < verticesNumber - 1; count++)
            {
                int u = MinKey(verticesNumber, key, mstSet);
                mstSet[u] = true;
                for (int v = 0; v < verticesNumber; v++)
                    if ((graphMatrix[u, v] != (-1)) && mstSet[v] == false
                                                    && graphMatrix[u, v] < key[v])
                    {
                        prev[v] = u;
                        key[v] = graphMatrix[u, v];
                    }
            }

            return prev;
        }

        static int ParallelMinKey(int verticesNumber, int[] key, bool[] mstSet, int chunkNum, int[] chunkSizes,
            AutoResetEvent allDoneMin)
        {
            int[] mins = new int[chunkNum];
            int[] minIndices = new int[chunkNum];
            for (int t = 0; t < chunkNum; t++)
            {
                mins[t] = Int32.MaxValue;
                minIndices[t] = -1;
            }

            int amount = 0;
            for (int t = 0; t < chunkNum; t++)
            {
                int chunkStart = chunkSizes[0] * t;
                int chunkEnd = chunkStart + chunkSizes[t];
                int tt = t;
                ThreadPool.QueueUserWorkItem((dumpObj) =>
                {
                    for (int v = chunkStart; v < chunkEnd; v++)
                    {
                        if (mstSet[v] == false && key[v] < mins[tt])
                        {
                            mins[tt] = key[v];
                            minIndices[tt] = v;
                        }
                    }

                    if (Interlocked.Increment(ref amount) == chunkNum)
                    {
                        allDoneMin.Set();
                    }
                }, tt);
            }
            allDoneMin.WaitOne();

            int min = Int32.MaxValue, minIndex = -1;
            for (int t = 0; t < chunkNum; t++)
            {
                if (mins[t] < min)
                {
                    min = mins[t];
                    minIndex = minIndices[t];
                }
            }
            
            return minIndex;
        }

        public static int[] RunParallelPrim(int[,] graphMatrix)
        {
            int verticesNumber = graphMatrix.GetLength(0);
            int[] prev = new int[verticesNumber];
            int[] key = new int[verticesNumber];
            bool[] mstSet = new bool[verticesNumber];

            for (int i = 0; i < verticesNumber; i++)
            {
                key[i] = int.MaxValue;
                mstSet[i] = false;
            }

            key[0] = 0;
            prev[0] = -1;
            int chunkNum = Environment.ProcessorCount;
            int[] chunkSizes = new int[chunkNum];
            for (int i = 0; i < chunkNum; i++)
            {
                chunkSizes[i] = verticesNumber / chunkNum;
            }

            chunkSizes[chunkNum - 1] += verticesNumber % chunkNum;

            AutoResetEvent allDone = new AutoResetEvent(false);
            AutoResetEvent allDoneMin = new AutoResetEvent(false);
            for (int count = 0; count < verticesNumber - 1; count++)
            {
                int u = ParallelMinKey(verticesNumber, key, mstSet, chunkNum, chunkSizes, allDoneMin);
                mstSet[u] = true;
                int amount = 0;
                for (int t = 0; t < chunkNum; t++)
                {
                    int chunkStart = chunkSizes[0] * t;
                    int chunkEnd = chunkStart + chunkSizes[t];
                    ThreadPool.QueueUserWorkItem((dumpObj) =>
                    {
                        for (int v = chunkStart; v < chunkEnd; v++)
                        {
                            if ((graphMatrix[u, v] != (-1))
                                && mstSet[v] == false
                                && graphMatrix[u, v] < key[v])
                            {
                                prev[v] = u;
                                key[v] = graphMatrix[u, v];
                            }
                        }

                        if (Interlocked.Increment(ref amount) == chunkNum)
                        {
                            allDone.Set();
                        }
                    });
                }

                allDone.WaitOne();
            }

            return prev;
        }
    }
}