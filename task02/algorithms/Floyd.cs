using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace Task02
{
    static class Floyd
    {
        public static Int32[,] FindShortestPaths(Int32 numberOfVertices, IEnumerable<Edge> edges)
        {
            var result = new Int32[numberOfVertices, numberOfVertices];

            for (Int32 i = 0; i < numberOfVertices; i++)
            {
                for (Int32 j = 0; j < numberOfVertices; j++)
                {
                    result[i, j] = -1;
                }
            }

            foreach (var edge in edges)
            {
                result[edge.first, edge.second] = edge.weight;
                result[edge.second, edge.first] = edge.weight;
            }

            for (Int32 i = 0; i < numberOfVertices; i++)
            {
                result[i, i] = 0;
            }

            AutoResetEvent allDone = new AutoResetEvent(false);
            Int32 processed = 0;

            for (Int32 k = 0; k < numberOfVertices; k++)
            {
                ThreadPool.QueueUserWorkItem(k =>
                {
                    for (Int32 i = 0; i < numberOfVertices; i++)
                    {
                        for (Int32 j = 0; j < numberOfVertices; j++)
                        {
                            var path_ik = result[i, (Int32)k];
                            var path_kj = result[(Int32)k, j];

                            if (path_ik < 0 || path_kj < 0)
                            {
                                continue;
                            }

                            if (result[i, j] < 0 || result[i, j] > path_ik + path_kj)
                            {
                                result[i, j] = path_ik + path_kj;
                            }
                        }
                    }

                    Interlocked.Increment(ref processed);
                    if (processed == numberOfVertices) {
                        allDone.Set();
                    }
                }, k);
            }

            allDone.WaitOne();

            return result;
        }
    }
}
