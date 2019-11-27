using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace Task02
{
    public static class Floyd
    {
        //this one with tasks (Parallel)
        public static int[,] ExecTaskFloyd(List<Edge> edges, int vertices)
        {
            int[,] dist = Helper.GetIncidenceMatrix(edges, vertices);

            for (int k = 0; k < vertices; k++)
            {
                //IDE suggested to make a local variable due to some closure issues
                //I suppose it ain't beneficial here
                //can read a bit about captured variables at https://jonskeet.uk/csharp/csharp2/delegates.html#captured.variables
                var k1 = k;
                Parallel.For(0, vertices, i =>
                {
                    for (var j = 0; j < vertices; ++j)
                        if (dist[i, k1] != It.Inf
                            && dist[k1, j] != It.Inf
                            && (dist[i, k1] + dist[k1, j]) < dist[i, j])
                        {
                            dist[i, j] = dist[i, k1] + dist[k1, j];
                        }
                });
            }
            
            return dist;
        }
        
        //this one with threads
        public static int[,] ExecFloyd(List<Edge> edges, int vertices)
        {
            int[,] dist = Helper.GetIncidenceMatrix(edges, vertices);

            int threadsNum = Math.Min(Environment.ProcessorCount, vertices);
            Thread[] threads = new Thread[threadsNum];
            //block corresponds to lines where [k] is chosen
            int blockSize = vertices / threadsNum;

            for (int threadIndex = 0; threadIndex < threadsNum; threadIndex ++)
            {
                int blockStart = threadIndex * blockSize;
                int blockEnd = (vertices - blockStart < 1.5 * blockSize) ? 
                    vertices : blockStart + blockSize;

                threads[threadIndex] = new Thread(() =>
                {
                    for (int k = blockStart; k < blockEnd; k++)
                    {
                        for (int i = 0; i < vertices; i++)
                        {
                            for (int j = 0; j < vertices; j++)
                            {
                                if (dist[i, k] != It.Inf
                                    && dist[k, j] != It.Inf
                                    && (dist[i, k] + dist[k, j]) < dist[i, j])
                                {
                                    dist[i, j] = dist[i, k] + dist[k, j];
                                }
                            }
                        }
                    }
                });
                
                threads[threadIndex].Start();
            }
            
            foreach (Thread t in threads) t.Join();

            return dist;
        }
    }
}