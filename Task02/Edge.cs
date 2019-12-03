using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Task02
{
    public class Edge : IComparable<Edge>
    {
        public Edge(int startVertex, int endVertex, int weight)
        {
            this.startVertex = startVertex;
            this.endVertex = endVertex;
            this.weight = weight;
        }

        public readonly int startVertex;
        public readonly int endVertex;
        public readonly int weight;

        public bool Equals(Edge other)
        {
            return startVertex == other.startVertex 
                && endVertex == other.endVertex;
        }

        public int CompareTo(Edge other)
        {
            return weight.CompareTo(other.weight);
        }
    }
}
