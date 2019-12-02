using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace Task02
{
    public class ParallelGraphSolver
    {
        public ParallelGraphSolver(int[,] graphMatrix, int numVertexes)
        {
            this.graphMatrix = graphMatrix;
            this.numVertexes = numVertexes;

            //Converting graph from matrix representation
            for (int i = 0; i < numVertexes; i++)
            {
                for (int j = 0; j < numVertexes; j++)
                {
                    if (graphMatrix[i, j] != Config.emptyEdge)
                    {
                        graphEdges.Add(new Edge(i, j, graphMatrix[i, j]));
                    }
                }
            }
        }

        public readonly int[,] graphMatrix;
        public readonly List<Edge> graphEdges = new List<Edge>();
        public readonly int numVertexes;
        
        public int[,] ParallelFloydSolve()
        {
            int[,] resultMatrix = new int[numVertexes, numVertexes];

            for (int k = 0; k < numVertexes; k++) {
                Parallel.For(0, numVertexes, i =>
                {
                    for (int j = 0; j < numVertexes; j++)
                    {
                        if (graphMatrix[k, j] != Config.emptyEdge
                            && graphMatrix[i, k] != Config.emptyEdge)
                        {
                            resultMatrix[i, j] = Math.Min(graphMatrix[i, j]
                                , graphMatrix[i, k] + graphMatrix[k, j]);
                        }
                        else
                        {
                            resultMatrix[i, j] = graphMatrix[i, j];
                        }
                    }
                });
            }

            return resultMatrix;
        }

        public int ParallelPrimSolve()
        {
            int result = 0;
            bool[] usedInMst = new bool[numVertexes];
            int[] minDistToMst = new int[numVertexes];

            for (int i = 0; i < numVertexes; i++)
            {
                minDistToMst[i] = int.MaxValue;
                usedInMst[i] = false;
            }

            //Including initial vertex into MST
            minDistToMst[0] = 0;
            usedInMst[0] = true;

            for (int count = 0; count < numVertexes; count++)
            {
                int minVertex = GetMinVertex(minDistToMst, usedInMst);
                usedInMst[minVertex] = true;
                result += minDistToMst[minVertex];
                
                int chunks = numVertexes / Config.chunkSize;

                using (var countdown = new CountdownEvent(chunks))
                {
                    for (int i = 0; i < chunks; i++)
                    {
                        int chunkStart = i * Config.chunkSize;
                        int chunkEnd = chunkStart + Config.chunkSize;

                        ThreadPool.QueueUserWorkItem(_ =>
                        {
                            for (int vertex = chunkStart; vertex < chunkEnd; vertex++)
                            {
                                if (graphMatrix[minVertex, vertex] != Config.emptyEdge
                                    && graphMatrix[minVertex, vertex] < minDistToMst[vertex]
                                    && !usedInMst[vertex])
                                {
                                    minDistToMst[vertex] = graphMatrix[minVertex, vertex]; ;
                                }
                            }
                            countdown.Signal();
                        });
                    }

                    countdown.Wait();
                }
            }

            return result;
        }
        
        public int ParallelKruskalSolve()
        {
            int result = 0;
            
            //Parallel quick sort using Tasks
            ParallelSort.QuickSort(graphEdges);

            //Placing each vertex into its own subset
            var subsetId = new int[numVertexes];
            for (int j = 0; j < numVertexes; j++)
            {
                subsetId[j] = j;
            }

            foreach (var edge in graphEdges)
            {
                int subId = subsetId[edge.startVertex];
                int adjacentSubId = subsetId[edge.endVertex];

                if (edge.weight != Config.emptyEdge 
                    && subId != adjacentSubId)
                {
                    result += edge.weight;
                    UpdateSubsets(subId, adjacentSubId, ref subsetId);
                }
            }

            return result;
        }
        
        private int GetMinVertex(int[] minDistToMst, bool[] usedInMst)
        {
            int minVertex = 0;
            int minDist = int.MaxValue;
            
            int chunks = numVertexes / Config.chunkSize;
            using (var countdown = new CountdownEvent(chunks))
            {
                for (int i = 0; i < chunks; i++)
                {
                    int chunkStart = i * Config.chunkSize;
                    int chunkEnd = chunkStart + Config.chunkSize;

                    ThreadPool.QueueUserWorkItem(_ =>
                    {
                        for (int vertex = chunkStart; vertex < chunkEnd; vertex++)
                        {
                            if (minDistToMst[vertex] < minDist
                                && !usedInMst[vertex])
                            {
                                minDist = minDistToMst[vertex];
                                minVertex = vertex;
                            }
                        }
                        countdown.Signal();
                    });
                }
                countdown.Wait();

                return minVertex;
            }
        }

        private void UpdateSubsets(int subId, int adjacentSubId, ref int[] subsetId)
        {
            for (int vertex = 0; vertex < numVertexes; vertex++)
            {
                if (subsetId[vertex] == subId)
                {
                    subsetId[vertex] = adjacentSubId;
                }
            }
        }
    }
}
