using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace Task_02
{
    public static class PrimaAlgo
    {
        private static int INF = int.MaxValue;
        private static int numOfThreads = Environment.ProcessorCount;

        internal static int SeqPrim1(List<Edge> graph, int countOfVertex)
        {
            var MST = new List<int>(graph.Count);
            var resWeight = 0;
            var notUsedE = new List<Edge>(graph);
            var usedV = new List<int>();
            var notUsedV = new List<int>();

            for (var i = 0; i < countOfVertex; i++) notUsedV.Add(i);
            
            var rand = new Random();
            usedV.Add(rand.Next(0, countOfVertex));
            notUsedV.RemoveAt(usedV[0]);
            
            while (notUsedE.Count > 0)
            {
                var minE = -1;

                    for (var i = 0; i < notUsedE.Count; i++)
                    {
                        //Console.WriteLine(Thread.CurrentThread.ManagedThreadId);
                        if ((usedV.IndexOf(notUsedE[i].From) == -1 || notUsedV.IndexOf(notUsedE[i].To) == -1) &&
                            (usedV.IndexOf(notUsedE[i].To) == -1 || notUsedV.IndexOf(notUsedE[i].From) == -1)) continue;
                        if (minE != -1)
                        {
                            if (notUsedE[i].Weight < notUsedE[minE].Weight)
                                minE = i;
                        }
                        else
                            minE = i;
                    }

                if (usedV.IndexOf(notUsedE[minE].From) != -1)
                {
                    usedV.Add(notUsedE[minE].To);
                    notUsedV.Remove(notUsedE[minE].To);
                }
                else
                {
                    usedV.Add(notUsedE[minE].From);
                    notUsedV.Remove(notUsedE[minE].From);
                }

                MST.Add(notUsedE[minE].Weight);
                resWeight += notUsedE[minE].Weight;
                notUsedE.RemoveAt(minE);
            }

            return resWeight;
        }

        internal static int SeqPrim2(Dictionary<int, List<int>> graph)
        {
            var vertices = graph.Count;
            var used = new bool[vertices];
            var minEdges = new int[vertices];
            //var selectedEdges = new int[vertices];
            
            //init
            for (var i = 0; i < vertices; i++)
            {
                minEdges[i] = INF;
                //selectedEdges[i] = -1;
            }

            minEdges[0] = 0;
            for (var i = 0; i < vertices; i++)
            {
                var vertex = -1;

                for (var j = 0; j < vertices; j++)
                {
                    if (!used[j] && (vertex == -1 || minEdges[j] < minEdges[vertex]))
                        vertex = j;
                }

                used[vertex] = true;
                //if (selectedEdges[vertex] != -1) Console.WriteLine($"{vertex} {selectedEdges[vertex]}");

                for (var to = 0; to < vertices; to++)
                {
                    if (graph[vertex][to] >= minEdges[to]) continue;
                    minEdges[to] = graph[vertex][to];
                    //selectedEdges[to] = vertex;
                }
                
            }

            return minEdges.Sum();
        }

        internal static int ParallelPrim(Dictionary<int, List<int>> graph, int countOfVertex)
        {
            var resWeight = 0;
            
            var vertices = graph.Count;
            var used = new bool[vertices];
            var minEdges = new int[vertices];
            
            var chunks = vertices / numOfThreads;
            var allDone = new ManualResetEvent(false);
            
            for (var i = 0; i < vertices; i++)
            {
                minEdges[i] = INF;
            }

            minEdges[0] = 0;
            for (var i = 0; i < vertices; i++)
            {
                var completed = 0;
                allDone.Reset();
                //var vertex = -1;
                var currentVertices = new int[chunks];
                for (var j = 0; j < chunks; j++)
                {
                    currentVertices[j] = -1;
                }
                
                for (var threadInd = 0; threadInd < chunks; threadInd++)
                {
                    var start = chunks * threadInd;
                    var end = chunks * (threadInd + 1);
                    if (vertices - start < chunks * 2)
                    {
                        end = vertices;
                    }

                    var j = threadInd;
                    ThreadPool.QueueUserWorkItem(_ =>
                    {
                        for (var vertex = start; vertex < end; vertex++)
                        {
                            if (!used[vertex] && (currentVertices[j] == -1 || minEdges[vertex] < minEdges[currentVertices[j]]))
                                currentVertices[j] = vertex;
                        }

                        if (Interlocked.Increment(ref completed) == chunks)
                        {
                            allDone.Set();
                        }
                    });
                }
                allDone.WaitOne();

                //choosing a minimal edge among selected above
                var current = -1;

                foreach (var min in currentVertices)
                {
                    if (min == -1) continue;
                    if (!used[min] && (current == -1 || minEdges[min] < minEdges[current]))
                        current = min;
                }

                if (current == -1) continue;

                if (minEdges[current] != INF)
                {
                    resWeight += minEdges[current];
                }
                used[current] = true;

                completed = 0;
                allDone.Reset();
                
                for (var threadInd = 0; threadInd < chunks; threadInd++)
                {
                    var start = chunks * threadInd;
                    var end = chunks * (threadInd + 1);
                    if (vertices - start < chunks * 2)
                    {
                        end = vertices;
                    }
                    ThreadPool.QueueUserWorkItem(_ =>
                    {
                        for (var to = start; to < end; to++)
                        {
                            if (graph[current][to] >= minEdges[to]) continue;
                            minEdges[to] = graph[current][to];
                        }
                        if (Interlocked.Increment(ref completed) == chunks)
                        {
                            allDone.Set();
                        }
                    });
                }
                allDone.WaitOne();
            }

            return minEdges.Sum();
        }


       
    }
}