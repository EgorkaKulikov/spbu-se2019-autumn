using System;

namespace Task02
{
    class Edge : IComparable<Edge>
    {
        #region Exception messages
        private string FromGTToExceptionMessage   = "Destination vertex number must be greater than source vertex number!";
        private string CostLTZeroExceptionMessage = "Edge cost must be positive and greater than 0";
        private string CostGTDefaultMaxValue      = $"Edge cost must be lesser than {Default.EdgeMaxCost}";
        #endregion

        public int From { get; private set; }
        public int To   { get; private set; }
        public int Cost { get; private set; }

        public Edge(int from, int to, int cost = Default.EdgeMinCost)
        {
            CheckInput(from, to, cost);

            From = from;
            To   = to;
            Cost = cost;
        }

        public Edge((int from, int to, int cost) edge) 
                        : this(edge.from, edge.to, edge.cost) { }

        //IComparable<Edge> implementation
        public int CompareTo(Edge other)
        {
            if (this.Cost < other.Cost) return -1;
            else
            if (this.Cost == other.Cost) return 0;
            else
                return 1; //if this.Cost > other.Cost
        }

        public override bool Equals(object obj)
        {
            return Equals(obj as Edge);
        }

        public bool Equals(Edge other)
        {
            return other != null && this.GetHashCode() == other.GetHashCode();
        }

        public override int GetHashCode()
        {
            unchecked
            {
                //Large prime numbers to avoid hashing collisions
                const int HashingBase = (int)2166136261;
                const int HashingMultiplier = 16777619;

                int hash = HashingBase;
                hash = (hash * HashingMultiplier) ^ From.GetHashCode();
                hash = (hash * HashingMultiplier) ^ Cost.GetHashCode();
                hash = (hash * HashingMultiplier) ^ To.GetHashCode();
                return hash;
            }
        }

        #region Operators overload
        public static bool operator >(Edge a, Edge b) =>
            a.CompareTo(b) == 1  ? true
                                 : false;

        public static bool operator <(Edge a, Edge b) =>
            a.CompareTo(b) == -1 ? true
                                 : false;

        public static bool operator ==(Edge a, Edge b) =>
            a.CompareTo(b) == 0  ? true
                                 : false;

        public static bool operator !=(Edge a, Edge b) =>
            a.CompareTo(b) != 0  ? true
                                 : false;
        #endregion

        private void CheckInput(int from, int to, int cost)
        {
            if (from >= to)
                throw new Exception(FromGTToExceptionMessage);

            if (cost <= 0)
                throw new Exception(CostLTZeroExceptionMessage);

            if (cost > Default.EdgeMaxCost)
                throw new Exception(CostGTDefaultMaxValue);
        }
    }
}
