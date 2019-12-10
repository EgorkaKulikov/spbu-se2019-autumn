using System.Collections.Generic;
using System;

namespace Task02
{
    public class Graph
    {
        public int[,] AdjacencyMatrix;
        public int VerticesCount;
        public List<Edge> EdgesList;

        public Graph(string fileName)
        {
            this.ReadGraph(fileName);
        }
        
        public class Edge : IComparable<Edge>
        {
            public readonly int End;
            public readonly int Start;
            public readonly int Weight;

            public Edge(in int start, in int end, in int weight)
            {
                Start = start;
                End = end;
                Weight = weight; 
            }

            public int CompareTo(Edge other)
            {
                return Weight.CompareTo(other.Weight);
            }
        }
    }
}