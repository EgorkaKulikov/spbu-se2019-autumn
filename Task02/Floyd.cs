using System;
using System.Threading;

namespace Task02
{
    class Floyd
    {
        public static int[,] RunSeqFloyd(int[,] graphMatrix)
        {
            int verticesNumber = graphMatrix.GetLength(0);

            int[,] result = graphMatrix;

            for (int i = 0; i < verticesNumber; i++)
            {
                for (int j = 0; j < verticesNumber; j++)
                {
                    if (result[i, j] == -1)
                    {
                        result[i, j] = Int32.MaxValue;
                    }
                }
            }

            for (int k = 0; k < verticesNumber; k++)
            {
                for (int i = 0; i < verticesNumber; i++)
                {
                    for (int j = 0; j < verticesNumber; j++)
                    {
                        if (result[i, k] < Int32.MaxValue && result[k, j] < Int32.MaxValue)
                        {
                            result[i, j] = Math.Min(result[i, j], result[i, k] + result[k, j]);
                        }
                    }
                }
            }
            return result;
        }

        public static int[,] RunParallelFloyd(int[,] graphMatrix)
        {
            int verticesNumber = graphMatrix.GetLength(0);

            int[,] result = graphMatrix;

            int threadNum = Environment.ProcessorCount;
            Thread[] threads = new Thread[threadNum];
            int chunkSize = verticesNumber / threadNum;

            for (int t = 0; t < threadNum; t++)
            {
                int chunkStart = chunkSize * t;
                int chunkEnd = chunkSize * (t + 1);
                if (verticesNumber - chunkStart < 2 * chunkSize)
                {
                    chunkEnd = verticesNumber;
                }

                threads[t] = new Thread(() =>
                {
                    for (int i = chunkStart; i < chunkEnd; i++)
                    {
                        for (int j = 0; j < verticesNumber; j++)
                        {
                            if (result[i, j] == -1)
                            {
                                Interlocked.Exchange(ref result[i, j], Int32.MaxValue);
                            }
                        }
                    }
                });
                threads[t].Start();
            }

            foreach (var thread in threads)
            {
                thread.Join();
            }

            for (int t = 0; t < threadNum; t++)
            {
                int chunkStart = chunkSize * t;
                int chunkEnd = chunkSize * (t + 1);
                if (verticesNumber - chunkStart < 2 * chunkSize)
                {
                    chunkEnd = verticesNumber;
                }

                threads[t] = new Thread(() =>
                {
                    for (int k = chunkStart; k < chunkEnd; k++)
                    {
                        for (int i = 0; i < verticesNumber; i++)
                        {
                            for (int j = 0; j < verticesNumber; j++)
                            {
                                if (result[i, k] < Int32.MaxValue && result[k, j] < Int32.MaxValue)
                                {
                                    result[i, j] = Math.Min(result[i, j], result[i, k] + result[k, j]);
                                }
                            }
                        }
                    }
                });
                threads[t].Start();
            }

            foreach (var thread in threads)
            {
                thread.Join();
            }
            
            return result;
        }
    }
}