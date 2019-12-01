using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace GraphParallelAlgorithms
{
    public static class Prim
    {
        public static long RunSequentialPrim(this Graph g)
        {
            var used = new HashSet<int>();
            var unused = new HashSet<int>();

            used.Add(0);

            for (var i = 1; i < g.VerticesNumber; ++i) unused.Add(i);

            long result = 0;

            while (unused.Count != 0)
            {
                var cur = new Graph.Edge(0, 0, long.MaxValue);

                foreach (var v in used)
                foreach (var u in unused)
                {
                    var cost = g.AdjacencyMatrix[v, u];
                    if (cost != 0 && cur.weight > cost) cur = new Graph.Edge(v, u, cost);
                }

                result += cur.weight;
                unused.Remove(cur.end);
                used.Add(cur.end);
            }

            return result;
        }

        public static long RunParallelForPrim(this Graph g)
        {
            var used = new HashSet<int>();
            var unused = new HashSet<int>();

            used.Add(0);

            for (var i = 1; i < g.VerticesNumber; ++i) unused.Add(i);

            long result = 0;

            while (unused.Count != 0)
            {
                var curs = new long[g.VerticesNumber];
                var ends = new int[g.VerticesNumber];

                Parallel.ForEach(used, v =>
                {
                    foreach (var u in unused)
                    {
                        var cost = g.AdjacencyMatrix[v, u];
                        if (cost != 0 && (curs[v] == 0 || curs[v] > cost))
                        {
                            curs[v] = cost;
                            ends[v] = u;
                        }
                    }
                });

                var cur = new Graph.Edge(0, 0, long.MaxValue);

                foreach (var v in used)
                    if (cur.weight > curs[v])
                        cur = new Graph.Edge(v, ends[v], curs[v]);

                result += cur.weight;
                unused.Remove(cur.end);
                used.Add(cur.end);
            }

            return result;
        }

        public static long RunParallelThreadsPrim(this Graph g)
        {
            var used = new HashSet<int>();
            var unused = new HashSet<int>();

            used.Add(0);

            for (var i = 1; i < g.VerticesNumber; ++i) unused.Add(i);

            long result = 0;

            while (unused.Count != 0)
            {
                var curs = new long[g.VerticesNumber];
                var ends = new int[g.VerticesNumber];

                var isDone = new AutoResetEvent(false);
                var completed = used.Count;

                foreach (var v in used)
                    ThreadPool.QueueUserWorkItem(_ =>
                    {
                        foreach (var u in unused)
                        {
                            var cost = g.AdjacencyMatrix[v, u];
                            if (cost != 0 && (curs[v] == 0 || curs[v] > cost))
                            {
                                curs[v] = cost;
                                ends[v] = u;
                            }
                        }

                        if (Interlocked.Decrement(ref completed) == 0) isDone.Set();
                    });

                isDone.WaitOne();

                var cur = new Graph.Edge(0, 0, long.MaxValue);

                foreach (var v in used)
                    if (cur.weight > curs[v])
                        cur = new Graph.Edge(v, ends[v], curs[v]);

                result += cur.weight;
                unused.Remove(cur.end);
                used.Add(cur.end);
            }

            return result;
        }
    }
}