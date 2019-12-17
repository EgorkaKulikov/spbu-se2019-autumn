using System;
using System.IO;
using System.Threading;

namespace Task02
{
    class Floyd
    {
        public int[,] floydAlg(int[,] graph, int size)
        {
            var results = (int[,])graph.Clone();
            int numThreads = Environment.ProcessorCount;
            Thread[] threads = new Thread[numThreads];
            int threadSize = size / numThreads;

            for (int k = 0; k < size; k++)
            {
                for (int i = 0; i < numThreads; i++)
                {
                    int threadStart = i * threadSize;
                    int threadEnd = threadStart + threadSize;
                    if (i == numThreads - 1)
                        threadEnd = size;

                    var tmp = k;
                    threads[i] = new Thread(() =>
                    {
                        for (int row = threadStart; row < threadEnd; row++)
                        {
                            for (int column = 0; column < size; column++)
                            {
                                if (results[row, tmp] != Int32.MaxValue &&
                                    results[tmp, column] != Int32.MaxValue)
                                {
                                    results[row, column] = Math.Min(results[row, column],
                                    results[row, tmp] + results[tmp, column]);
                                }
                            }
                        }
                    });
                    threads[i].Start();
                }
                foreach (var thread in threads)
                    thread.Join();
            }

            return results;
        }

        public void printResult(int[,] graph, int size)
        {
            using (var sw = new StreamWriter("Floyd.txt"))
            {
                var result = floydAlg(graph, size);

                for (int i = 0; i < size; i++)
                {
                    for (int j = 0; j < size; j++)
                    {
                        sw.Write(result[i, j] + " ");
                    }

                    sw.WriteLine();
                }
            }
        }
    }
}
