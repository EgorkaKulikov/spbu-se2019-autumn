using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace Graphs
{
    public class ParallelAlgorithms
    {
        private readonly Graph _graphInit = new Graph();
        
        public int[,] SolveFloyd(int[,] graph, int size)
        {
            for (int row = 0; row < size; ++row)
            {
                graph[row, row] = 0;
            }
            
            int chunks = Environment.ProcessorCount;
            int chunkSize = size / chunks;
            
            WaitHandle[] waitHandles = new WaitHandle[chunks];

            for (int k = 0; k < size; ++k)
            {
                for (int chunk = 0; chunk < chunks; ++chunk)
                {
                    int chunkStart = chunk * chunkSize;
                    int chunkEnd = chunkStart + chunkSize;
                    if (chunk == chunks - 1)
                    {
                        chunkEnd = chunkStart + chunkSize + size % chunkSize;
                    }
                        
                    var handler = new AutoResetEvent(false);

                    var pivot = k;
                    var thread= new Thread(() =>
                    {
                        for (int row = chunkStart; row < chunkEnd; ++row)
                        {
                            for (int column = row + 1; column < chunkEnd; ++column)
                            {
                                if (graph[row, pivot] != Constants.Inf
                                    && graph[pivot, column] != Constants.Inf)
                                {
                                    graph[row, column] = Math.Min(graph[row, column]
                                        , graph[row, pivot] + graph[pivot, column]);

                                }
                            }
                        }

                        handler.Set();
                    });

                    waitHandles[chunk] = handler;
                    thread.Start();
                }

                WaitHandle.WaitAll(waitHandles);   
            }
            
            return graph;
        }

        public (int, List<Edge>) SolvePrim(int[,] graph, int size)
        {
            int weightOfTree = 0;
            List<Edge> mst = new List<Edge>();

            int[] minEdge = new int[size];
            int[] selEdge = new int[size];
            bool[] used = new bool[size];

            for (int i = 0; i < size; ++i)
            {
                minEdge[i] = Constants.Inf;
                selEdge[i] = -1;
                used[i] = false;
            }

            minEdge[0] = 0;
            
            int chunks = Environment.ProcessorCount;
            int chunkSize = size / chunks;
            AutoResetEvent allDone = new AutoResetEvent(false);
            
            for (int i = 0; i < size; ++i)
            {
                int indexLowestWeight = -1;
                int completed = 0;

                int[] indexesMin = new int[chunks];
                for (int index = 0; index < chunks; ++index)
                {
                    indexesMin[index] = -1;
                }

                for (int chunk = 0; chunk < chunks; ++chunk)
                {
                    int chunkStart = chunk * chunkSize;
                    int chunkEnd = chunkStart + chunkSize;
                    if (chunk == chunks - 1)
                    { 
                        chunkEnd = chunkStart + chunkSize + size % chunkSize;
                    }

                    var localChunk = chunk;
                    ThreadPool.QueueUserWorkItem(_ =>
                    {
                        for (int vert = chunkStart; vert < chunkEnd; ++vert)
                        {
                            if (!used[vert] && (indexesMin[localChunk] == -1 
                                             || (minEdge[vert] < minEdge[indexesMin[localChunk]])))
                            {
                                indexesMin[localChunk] = vert;
                            }
                        }

                        if (Interlocked.Increment(ref completed) == chunks)
                        {
                            allDone.Set();
                        }
                    });
                }
                allDone.WaitOne();

                foreach (var index in indexesMin)
                {
                    if (index != -1 && (indexLowestWeight == -1 
                                        || (minEdge[index] < minEdge[indexLowestWeight])))
                    {
                        indexLowestWeight = index;
                    }
                }
                
                if (minEdge[indexLowestWeight] == Constants.Inf)
                {
                    minEdge[indexLowestWeight] = 0;
                }
                
                weightOfTree += minEdge[indexLowestWeight];

                used[indexLowestWeight] = true;

                completed = 0;
                for (int chunk = 0; chunk < chunks; ++chunk)
                {
                    int chunkStart = chunk * chunkSize;
                    int chunkEnd = chunkStart + chunkSize;
                    if (chunk == chunks - 1)
                    { 
                        chunkEnd = chunkStart + chunkSize + size % chunkSize;
                    }

                    ThreadPool.QueueUserWorkItem(_ =>
                    {
                        for (int to = chunkStart; to < chunkEnd; ++to)
                        {
                            if (graph[indexLowestWeight, to] < minEdge[to])
                            {
                                minEdge[to] = graph[indexLowestWeight, to];
                                selEdge[to] = indexLowestWeight;
                            }
                        }
                        
                        if (Interlocked.Increment(ref completed) == chunks)
                        {
                            allDone.Set();
                        }
                    });
                }
                allDone.WaitOne();
            }

            for (int indexEdge = 0; indexEdge < size; ++indexEdge)
            {
                if (selEdge[indexEdge] != -1)
                { 
                    mst.Add(new Edge(indexEdge, selEdge[indexEdge], minEdge[indexEdge])); 
                }            
            }
            
            return (weightOfTree, mst);
        }

        public (int, List<Edge>) SolveKruskal(int[,] graph, int size)
        {
            List<Edge> newGraph = _graphInit.FromMatrixToListEdges(graph, size);
            int[] treeId = new int[size];
            List<Edge> mst = new List<Edge>();
            
            Edge[] newGr = newGraph.ToArray();
            ParallelSort.StartQuickSort(newGr);
            newGraph = newGr.ToList();

            int weightOfTree = 0;

            for (int i = 0; i < size; ++i)
            {
                treeId[i] = i;
            }

            foreach (var edge in newGraph)
            {
                int firstNode = edge.begin;
                int secondNode = edge.end;
                int weight = edge.weight;

                if (treeId[firstNode] != treeId[secondNode])
                {
                    weightOfTree += weight;
                    mst.Add(new Edge(firstNode, secondNode, weight));

                    int firstSubTree = treeId[firstNode];
                    int secondSubTree = treeId[secondNode];

                    for (int indexNode = 0; indexNode < size; ++indexNode)
                    {
                        if (treeId[indexNode] == secondSubTree)
                        {
                            treeId[indexNode] = firstSubTree;
                        }
                    }
                }
            }

            return (weightOfTree, mst);
        }
    }
}