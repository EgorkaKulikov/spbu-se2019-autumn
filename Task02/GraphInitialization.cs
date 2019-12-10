using System;
using System.IO;
using System.Collections.Generic;

namespace Task02
{
    public static class GraphInitialization
    {
        public static void ReadGraph(this Graph graph, string filename)
        {
            using var reader = new StreamReader(filename);
            graph.VerticesCount = int.Parse(reader.ReadLine());
            graph.AdjacencyMatrix = new int[graph.VerticesCount, graph.VerticesCount];
            graph.EdgesList = new List<Graph.Edge>();

            for (int i = 0; i < graph.VerticesCount; i++)
            {
                for (int j = 0; j < graph.VerticesCount; j++)
                {
                    if (i == j)
                    {
                        graph.AdjacencyMatrix[i, j] = 0;
                    }
                    else
                    {
                        graph.AdjacencyMatrix[i, j] = Int32.MaxValue;
                    }
                }
            }

            while (!reader.EndOfStream)
            {
                var edge = reader.ReadLine().Split();
                int start = int.Parse(edge[0]);
                int end = int.Parse(edge[1]);
                int weight = int.Parse(edge[2]);

                graph.EdgesList.Add(new Graph.Edge(start, end, weight));
                graph.AdjacencyMatrix[start, end] = weight;
                graph.AdjacencyMatrix[end, start] = weight;
            }
        }
    }
}