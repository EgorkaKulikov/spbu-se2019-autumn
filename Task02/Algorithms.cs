using System;
using System.Collections.Generic;
using System.Linq;

namespace Graphs
{
    public class Algorithms
    {
        private readonly Graph _graphInit = new Graph();
        
        public int[,] SolveFloyd(int[,] graph, int size)
        {
            for (int row = 0; row < size; ++row)
            {
                graph[row, row] = 0;
            }
            
            for (int k = 0; k < size; ++k)
            {
                for (int row = 0; row < size; ++row)
                {
                    for (int column = 0; column < size; ++column)
                    {
                        if (graph[row, k] != Constants.Inf 
                            && graph[k, column] != Constants.Inf)
                        {
                            graph[row, column] = Math.Min(graph[row, column], graph[row, k] + graph[k, column]);
                        }
                    }
                }
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
            
            for (int i = 0; i < size; ++i)
            {
                int indexLowestWeight = -1;
                for (int j = 0; j < size; ++j)
                {
                    if (!used[j] && (indexLowestWeight == -1 
                                     || (minEdge[j] < minEdge[indexLowestWeight])))
                    {
                        indexLowestWeight = j;
                    }
                }

                if (minEdge[indexLowestWeight] == Constants.Inf)
                {
                    minEdge[indexLowestWeight] = 0;
                }
                
                weightOfTree += minEdge[indexLowestWeight];

                used[indexLowestWeight] = true;

                for (int to = 0; to < size; ++to)
                {
                    if (graph[indexLowestWeight, to] < minEdge[to])
                    {
                        minEdge[to] = graph[indexLowestWeight, to];
                        selEdge[to] = indexLowestWeight;
                    }
                }
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
            SeqSort.StartQuickSort(newGr);
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