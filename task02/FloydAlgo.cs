
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace Task_02

{
    public static class FloydAlgo
    { 
        private static int INF = int.MaxValue;
        private static int numOfThreads = Environment.ProcessorCount;
        
    internal static Dictionary<int, List<int>> BuildFloydConsistent(Dictionary<int, List<int>> matrix)
    {
        var n = matrix.Count;
        var d = matrix;

        for (var k = 0; k < n; k++)
        {
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                { 
                    if (d[i][k] < INF && d[k][j] < INF)
                        d[i][j] = Math.Min(d[i][j], d[i][k] + d[k][j]);
                }
            }
        }

        return d;
    }
    
    internal static Dictionary<int, List<int>> BuildFloydCParallel(Dictionary<int, List<int>> matrix)
    {
        var vertex = matrix.Count;
        var result = matrix;
        var chunks = vertex / numOfThreads;
        var threads = new Thread[numOfThreads];

        for (var threadInd = 0; threadInd < numOfThreads; threadInd++)
        {
            var start = chunks * threadInd;
            var end = chunks * (threadInd + 1);
            if (vertex - start < chunks * 2)
            {
                end = vertex;
            }
            
            threads[threadInd] = new Thread(() =>
            {
                for (var k = start; k < end; k++)
                {
                    for (var i = 0; i < vertex; i++)
                    {
                        for (var j = 0; j < vertex; j++)
                        {
                            if (result[i][k] < INF && result[k][j] < INF)
                                result[i][j] = Math.Min(result[i][j], result[i][k] + result[k][j]);
                        }
                    }
                }
            });
            threads[threadInd].Start();
                    
        }
        
        foreach (var thread in threads) thread.Join();
        return result;
    }
    }
}
