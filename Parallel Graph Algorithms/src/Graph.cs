using System;
using System.Collections.Generic;

namespace GraphParallelAlgorithms
{
    public class Graph
    {
        public long[,] AdjacencyMatrix;
        public List<Edge> Edges;
        public int EdgesNumber;
        public int VerticesNumber;

        public Graph(string fileName)
        {
            this.ReadGraph(fileName);
        }

        public Graph()
        {
            Console.Write("Input vertices number V = ");
            VerticesNumber = Convert.ToInt32(Console.ReadLine());

            Console.Write("Input edges number E = ");
            EdgesNumber = Convert.ToInt32(Console.ReadLine());

            var random = new Random();

            Edges = new List<Edge>();
            AdjacencyMatrix = new long[VerticesNumber, VerticesNumber];

            while (EdgesNumber != Edges.Count)
            {
                int start = random.Next(0, VerticesNumber - 1),
                    end = random.Next(start + 1, VerticesNumber),
                    weight = random.Next(1, 100);

                if (AdjacencyMatrix[start, end] != 0) continue;

                Edges.Add(new Edge(start, end, weight));
                AdjacencyMatrix[start, end] = weight;
                AdjacencyMatrix[end, start] = weight;
            }
        }

        public class Edge : IComparable<Edge>
        {
            public int end;
            public int start;
            public long weight;

            public Edge() : this(0, 0, 0)
            {
            }

            public Edge(int start, int end, long weight)
            {
                this.start = start;
                this.end = end;
                this.weight = weight;
            }

            public int CompareTo(Edge other)
            {
                if (weight < other.weight) return -1;
                if (weight > other.weight) return 1;
                return 0;
            }
        }
    }
}