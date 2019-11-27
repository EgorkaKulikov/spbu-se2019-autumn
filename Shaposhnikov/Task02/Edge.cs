using System;

namespace Task02
{
    public class Edge : IComparable<Edge>
    {
            public readonly int First;
            public readonly int Second;
            public readonly int Weight;

            public Edge(int fst, int snd, int weight)
            {
                First = fst;
                Second = snd;
                Weight = weight;
            }
            
            public int CompareTo(Edge compareEdge) 
            { 
                return this.Weight - compareEdge.Weight; 
            } 
        }
}