using System;
using System.Collections.Generic;
using System.IO;

namespace Task02
{
    class Graph
    {
        public int[,] adjacencyMatrix;
        public int vertexCnt;
        public List<Edge> edges;
        public void ReadGraph(string filename)
        {
            using var reader = new StreamReader(filename);
            vertexCnt = int.Parse(reader.ReadLine());

            adjacencyMatrix = new int[vertexCnt, vertexCnt];
            for (int i = 0; i < vertexCnt; ++i)
                for (int j = 0; j < vertexCnt; ++j)
                    adjacencyMatrix[i, j] = (i == j) ? 0 : int.MaxValue;

            edges = new List<Edge>();

            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine().Split();
                int from = int.Parse(line[0]);
                int to = int.Parse(line[1]);
                int weight = int.Parse(line[2]);
                edges.Add(new Edge(from, to, weight));
                edges.Add(new Edge(to, from, weight));
                adjacencyMatrix[from, to] = weight;
                adjacencyMatrix[to, from] = weight;
            }
        }

        public void Generate(int vertices, int edges)
        {
            adjacencyMatrix = new int[vertices, vertices];

            Random rand = new Random();
            for (int i = 0; i < edges; i++)
            {
                int from, to;
                int weight = rand.Next(int.MaxValue);
                Edge newEdge, newEdgeBack;

                do
                {
                    from = rand.Next(edges - 1);
                    to = rand.Next(from + 1, edges);

                    newEdge = new Edge(from, to, weight);
                    newEdgeBack = new Edge(to, from, weight);
                } while (this.edges.Contains(newEdge));

                adjacencyMatrix[from, to] = weight;
                adjacencyMatrix[to, from] = weight;
                this.edges.Add(newEdge);
                this.edges.Add(newEdgeBack);
            }
        }
        public class Edge : IComparable<Edge>
        {
            public readonly int from, to, weight;

            public Edge(int from, int to, int weight)
            {
                this.from = from;
                this.to = to;
                this.weight = weight;
            }

            public int CompareTo(Edge other)
            {
                return weight - other.weight;
            }
        }
    }
}

