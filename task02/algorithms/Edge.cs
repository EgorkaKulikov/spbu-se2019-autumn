using System;

namespace Task02
{
    public class Edge : IComparable<Edge>
    {
        public Edge(Int32 first, Int32 second, Int32 weight)
        {
            this.first = first;
            this.second = second;
            this.weight = weight;
        }

        public readonly Int32 first;
        public readonly Int32 second;
        public readonly Int32 weight;

        public Int32 CompareTo(Edge edge) => weight.CompareTo(edge.weight);

        public Int32 OtherThan(Int32 index)
        {
            if (index == first)
            {
                return second;
            }

            if (index == second)
            {
                return first;
            }

            return -1;
        }
    }
}
