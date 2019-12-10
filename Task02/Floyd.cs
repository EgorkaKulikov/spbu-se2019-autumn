using System;
using System.IO;
using System.Threading;

namespace Task02
{
    public static class Floyd
    {
        private static int[,] ParallelFloyd(this Graph graph)
        {
            var result = (int[,])graph.AdjacencyMatrix.Clone();
            Thread[] threads = new Thread[graph.VerticesCount];
            
            for (int k = 0; k < graph.VerticesCount; k++)
            {
                for (int i = 0; i < graph.VerticesCount; i++)
                {
                    var i1 = i;
                    var k1 = k;
                    threads[i] = new Thread(() =>
                    {
                        for (int j = 0; j < graph.VerticesCount; j++)
                        {
                            if (result[i1, k1] < Int32.MaxValue && result[k1, j] < Int32.MaxValue)
                            {
                                result[i1, j] = Math.Min(result[i1, j], result[i1, k1] + result[k1, j]);
                            }
                        }
                    });
                    threads[i].Start();
                }
                foreach (var thread in threads)
                {
                    thread.Join();
                }
            }

            return result;
        }
        
        public static void PrintFloyd(this Graph graph)
        {
            using var writer = new StreamWriter("Floyd.txt");
            var result = graph.ParallelFloyd();

            for (var i = 0; i < graph.VerticesCount; i++)
            {
                for (var j = 0; j < graph.VerticesCount; j++)
                {
                    writer.Write($"{result[i, j]} ");
                }

                writer.WriteLine();
            }
        }
    }
}