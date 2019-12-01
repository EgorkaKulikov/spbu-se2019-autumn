using System;

namespace Task_02
{
    public class Edge : IComparable<Edge>
    {
        public int From { get; }
        public int To { get;}
        internal int Weight { get; set; }

        public Edge (int from, int to)
        {
            From = from;
            To = to;
            Weight = int.MaxValue;
        }

        public int CompareTo(Edge e)
        {
            return Weight.CompareTo(e.Weight);
        }

    }


}