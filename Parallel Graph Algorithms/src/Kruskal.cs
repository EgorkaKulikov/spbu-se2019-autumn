using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace GraphParallelAlgorithms
{
    public static class Kruskal
    {
        public static long RunSequentialKruskal(this Graph g)
        {
            var dsu = new DisjointSetUnion(g.VerticesNumber);
            var edges = g.Edges.ToList();
            edges.Sort();

            long result = 0;

            foreach (var edge in edges)
                if (dsu.Find(edge.start) != dsu.Find(edge.end))
                {
                    result += edge.weight;
                    dsu.Union(edge.start, edge.end);
                }

            return result;
        }

        public static long RunParallelForKruskal(this Graph g)
        {
            var dsu = new DisjointSetUnion(g.VerticesNumber);
            var edges = new List<Graph.Edge>();

            var chunkSize = g.EdgesNumber;

            if (g.EdgesNumber > Environment.ProcessorCount) chunkSize = g.EdgesNumber / Environment.ProcessorCount;

            var chunksNumber = g.EdgesNumber / chunkSize;

            var chunks = new List<Graph.Edge>[chunksNumber];
            var start = new int[chunksNumber];

            for (int i = 0, chunk = chunkSize; i < g.EdgesNumber; i += chunk)
            {
                if (i + 2 * chunk > g.EdgesNumber)
                    chunk = g.EdgesNumber - i;

                chunks[i / chunkSize] = new List<Graph.Edge>();

                for (var j = i; j < i + chunk; ++j) chunks[i / chunkSize].Add(g.Edges[j]);
            }

            Parallel.For(0, chunks.Length, i => { chunks[i].Sort(); });

            while (edges.Count != g.EdgesNumber)
            {
                var cur = new Graph.Edge();
                var id = -1;

                for (var i = 0; i < chunks.Length; ++i)
                    if (start[i] < chunks[i].Count && (id == -1 || chunks[i][start[i]].weight < cur.weight))
                    {
                        id = i;
                        cur = chunks[i][start[i]];
                    }

                edges.Add(cur);
                ++start[id];
            }

            long result = 0;

            foreach (var edge in edges)
                if (dsu.Find(edge.start) != dsu.Find(edge.end))
                {
                    result += edge.weight;
                    dsu.Union(edge.start, edge.end);
                }

            return result;
        }

        public static long RunParallelThreadsKruskal(this Graph g)
        {
            var dsu = new DisjointSetUnion(g.VerticesNumber);
            var edges = new List<Graph.Edge>();

            var chunkSize = g.EdgesNumber;

            if (g.EdgesNumber > Environment.ProcessorCount) chunkSize = g.EdgesNumber / Environment.ProcessorCount;

            var chunksNumber = g.EdgesNumber / chunkSize;

            var chunks = new List<Graph.Edge>[chunksNumber];
            var tasks = new List<Task>();
            var start = new int[chunksNumber];

            for (int i = 0, chunk = chunkSize; i < g.EdgesNumber; i += chunk)
            {
                if (i + 2 * chunk > g.EdgesNumber)
                    chunk = g.EdgesNumber - i;

                chunks[i / chunkSize] = new List<Graph.Edge>();

                for (var j = i; j < i + chunk; ++j) chunks[i / chunkSize].Add(g.Edges[j]);

                var i1 = i / chunkSize;
                tasks.Add(Task.Run(() => chunks[i1].Sort()));
            }

            Task.WaitAll(tasks.ToArray());

            while (edges.Count != g.EdgesNumber)
            {
                var cur = new Graph.Edge();
                var id = -1;

                for (var i = 0; i < chunks.Length; ++i)
                    if (start[i] < chunks[i].Count && (id == -1 || chunks[i][start[i]].weight < cur.weight))
                    {
                        id = i;
                        cur = chunks[i][start[i]];
                    }

                edges.Add(cur);
                ++start[id];
            }

            long result = 0;

            foreach (var edge in edges)
                if (dsu.Find(edge.start) != dsu.Find(edge.end))
                {
                    result += edge.weight;
                    dsu.Union(edge.start, edge.end);
                }

            return result;
        }

        private class DisjointSetUnion
        {
            private readonly int[] _parent;
            private readonly int[] _rank;

            public DisjointSetUnion(int verticesNumber)
            {
                _parent = new int[verticesNumber];
                _rank = new int[verticesNumber];

                for (var i = 0; i < verticesNumber; ++i)
                {
                    _parent[i] = i;
                    _rank[i] = 0;
                }
            }

            public int Find(int v)
            {
                if (v == _parent[v]) return v;
                _parent[v] = Find(_parent[v]);
                return _parent[v];
            }

            public void Union(int v, int u)
            {
                v = Find(v);
                u = Find(u);

                if (v == u) return;

                if (_rank[v] < _rank[u])
                {
                    v += u;
                    u = v - u;
                    v -= u;
                }

                if (_rank[v] == _rank[u]) ++_rank[v];

                _parent[u] = v;
            }
        }
    }
}