using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Task02
{
    class Algorithms
    {
        public class Floyd
        {
            public static int[,] SeqSolve(int[,] graph, int vertexCnt)
            {
                for (int k = 0; k < vertexCnt; ++k)
                    for (int i = 0; i < vertexCnt; ++i)
                        for (int j = 0; j < vertexCnt; ++j)
                            if (graph[i, k] != int.MaxValue && graph[k, j] != int.MaxValue)
                                graph[i, j] = Math.Min(graph[i, j], graph[i, k] + graph[k, j]);
                return graph;
            }

            public static int[,] ParSolve(int[,] graph, int vertexCnt)
            {
                Thread[] threads = new Thread[vertexCnt];
                for (int v = 0; v < vertexCnt; v++)
                {
                    for (int index = 0; index < vertexCnt; index++)
                    {
                        var i = index;
                        var k = v;
                        threads[index] = new Thread(() =>
                        {
                            for (int j = 0; j < vertexCnt; j++)
                                if (graph[i, k] != int.MaxValue && graph[k, j] != int.MaxValue)
                                    graph[i, j] = Math.Min(graph[i, j], graph[i, k] + graph[k, j]);
                        });
                        threads[index].Start();
                    }
                    foreach (var thread in threads)
                    {
                        thread.Join();
                    }
                }
                return graph;
            }
        }

        public class Kruskal
        {
            private class DisjointSetUnion
            {
                public int[] parent, rank;

                public DisjointSetUnion(int vertexCnt)
                {
                    parent = new int[vertexCnt];
                    rank = new int[vertexCnt];
                    for (int i = 0; i < vertexCnt; ++i)
                    {
                        parent[i] = i;
                        rank[i] = 0;
                    }
                }

                public int Find(int v)
                {
                    if (v == parent[v])
                        return v;
                    return parent[v] = Find(parent[v]);
                }

                public void Union(int a, int b)
                {
                    a = parent[a];
                    b = parent[b];
                    if (a == b)
                        return;

                    if (rank[a] > rank[b])
                        parent[b] = a;
                    else
                    {
                        parent[a] = b;
                        if (rank[a] == rank[b])
                            ++rank[b];
                    }
                }
            }

            public static int ParSolve(Graph.Edge[] edges, int vertexCnt)
            {
                int result = 0;
                var dsu = new DisjointSetUnion(vertexCnt);
                Sort.ParallelQSort(edges, 0, edges.Length);
                int i = 0, cnt = 0;
                while (cnt < vertexCnt - 1)
                {
                    var edge = edges[i++];

                    int a = dsu.Find(edge.to);
                    int b = dsu.Find(edge.from);

                    if (a != b)
                    {
                        edges[cnt++] = edge;
                        dsu.Union(a, b);
                        result += edge.weight;
                    }
                }
                return result;
            }

            public static int SeqSolve(Graph.Edge[] edges, int vertexCnt)
            {
                int result = 0;
                var dsu = new DisjointSetUnion(vertexCnt);
                Sort.QSort(edges, 0, edges.Length);
                int i = 0, cnt = 0;
                while (cnt < vertexCnt - 1)
                {
                    var edge = edges[i++];

                    int a = dsu.Find(edge.to);
                    int b = dsu.Find(edge.from);

                    if (a != b)
                    {
                        edges[cnt++] = edge;
                        dsu.Union(a, b);
                        result += edge.weight;
                    }
                }
                return result;
            }

            private class Sort
            {
                private static void Swap(ref Graph.Edge i, ref Graph.Edge j)
                {
                    var temp = i;
                    i = j;
                    j = temp;
                }

                private static int Partition(Graph.Edge[] array, int left, int right)
                {
                    var m = array[(left + right) / 2];
                    Swap(ref array[right - 1], ref array[(left + right) / 2]);
                    int pivot = left;

                    for (int i = left; i < right - 1; ++i)
                    {
                        if (array[i].CompareTo(m) < 0)
                        {
                            Swap(ref array[i], ref array[pivot]);
                            pivot++;
                        }
                    }

                    Swap(ref array[right - 1], ref array[pivot]);
                    return pivot;
                }
                public static void ParallelQSort(Graph.Edge[] array, int left, int right)
                {
                    if (left >= right)
                        return;

                    int pivot = Partition(array, left, right);

                    Task sortLeft = Task.Run(() => QSort(array, left, pivot));
                    Task sortRight = Task.Run(() => QSort(array, pivot + 1, right));
                    Task.WaitAll(sortLeft, sortRight);
                }

                public static void QSort(Graph.Edge[] array, int left, int right)
                {
                    if (left >= right)
                        return;

                    int pivot = Partition(array, left, right);

                    QSort(array, left, pivot);
                    QSort(array, pivot + 1, right);
                }
            }
            
        }

        public class Prim
        {
            public static int ParSolve(int[,] graph, int vertexCnt)
            {
                int result = 0;
                int[] minWeight = Enumerable.Repeat(int.MaxValue, vertexCnt).ToArray();
                int[] minEdgeFrom = Enumerable.Repeat(-1, vertexCnt).ToArray();
                bool[] used = Enumerable.Repeat(false, vertexCnt).ToArray();

                minWeight[0] = 0;
                for (int i = 0; i < vertexCnt; ++i)
                {
                    int v = -1;
                    for (int j = 0; j < vertexCnt; ++j)
                        if (!used[j] && (v == -1 || minWeight[j] < minWeight[v]))
                            v = j;

                    used[v] = true;
                    result += minWeight[v];

                    int completedCnt = 0;
                    var allCompleted = new ManualResetEvent(false);
                    for (int j = 0; j < vertexCnt; ++j)
                    {
                        int u = j;
                        ThreadPool.QueueUserWorkItem(_ =>
                        {
                            if (!used[u] && graph[u, v] < minWeight[u])
                            {
                                minWeight[u] = graph[u, v];
                                minEdgeFrom[v] = u;
                            }
                            if (Interlocked.Increment(ref completedCnt) == vertexCnt)
                                allCompleted.Set();
                        });
                    }
                    allCompleted.WaitOne();

                }
                return result;
            }

            public static int SeqSolve(int[,] graph, int vertexCnt)
            {
                int result = 0;
                int[] minWeight = Enumerable.Repeat(int.MaxValue, vertexCnt).ToArray();
                int[] minEdgeFrom = Enumerable.Repeat(-1, vertexCnt).ToArray();
                bool[] used = Enumerable.Repeat(false, vertexCnt).ToArray();

                minWeight[0] = 0;
                for (int i = 0; i < vertexCnt; ++i)
                {
                    int v = -1;
                    for (int j = 0; j < vertexCnt; ++j)
                        if (!used[j] && (v == -1 || minWeight[j] < minWeight[v]))
                            v = j;

                    used[v] = true;
                    result += minWeight[v];

                    for (int j = 0; j < vertexCnt; ++j)
                    {
                        int u = j;
                        if (!used[u] && graph[u, v] < minWeight[u])
                        {
                            minWeight[u] = graph[u, v];
                            minEdgeFrom[v] = u;
                        }
                    }
                }
                return result;
            }
        }
    }
}
