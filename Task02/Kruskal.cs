using System.Collections.Generic;
using System.Threading.Tasks;

namespace Task02
{
    static class Kruskal
    {
        #region Utils
        private static Dictionary<int, int> source = new Dictionary<int, int>();
        private static Dictionary<int, int> rank = new Dictionary<int, int>();

        private static void InitDictionaries(int num)
        {
            for (int k = 0; k < num; k++)
            {
                source[k] = k;
                rank[k] = 0;
            }
        }

        private static int GetSource(int k)
        {
            if (k == source[k])
                return k;
            else
                return source[k] = GetSource(source[k]);
        }

        private static void SwapRef(ref int a, ref int b)
        {
            var temp = a;
            a = b;
            b = temp;
        }

        private static void Link(int a, int b)
        {
            a = GetSource(a); b = GetSource(b);

            if (!Equals(a, b))
            {
                if (rank[a] < rank[b])
                    SwapRef(ref a, ref b);

                source[b] = a;

                if (rank[a] == rank[b])
                    rank[b]++;
            }
        }
        #endregion

        #region Sort
        private static void SortAsync(Edge[] edges, int left, int right)
        {
            if (right - left <= 1)
                return;

            int pivot = Partition(edges, left, right);

            if (right - left > Default.ParallelSortThreshold)
            {
                Task.WaitAll(
                                Task.Run(() => SortAsync(edges, left, pivot)),
                                Task.Run(() => SortAsync(edges, pivot + 1, right))
                            );
            }
            else
            {
                Sort(edges, left, pivot);
                Sort(edges, pivot + 1, right);
            }
        }

        private static int Partition(Edge[] edges, int left, int right)
        {
            Edge x = edges[left];
            int i = left - 1;
            int j = right + 1;
            while (true)
            {
                do j--; while (edges[j] > x);
                do i++; while (edges[i] < x);

                if (i < j)
                {
                    Edge tmp = edges[i];
                    edges[i] = edges[j];
                    edges[j] = tmp;
                }
                else
                    return j;
            }
        }

        private static void Sort(Edge[] edges, int left, int right)
        {
            if (left == right) return;
            int pivot = Partition(edges, left, right);
            Sort(edges, left, pivot);
            Sort(edges, pivot + 1, right);
        }
        #endregion

        public static int Run(Graph graph)
        {
            int verticesNum = graph.VerticesNum;
            var edges = graph.GetEdges().ToArray();
            SortAsync(edges, 0, edges.Length - 1);

            InitDictionaries(verticesNum);

            int result = 0;
            foreach (var edge in edges)
            {
                if (GetSource(edge.From) != GetSource(edge.To))
                {
                    result += edge.Cost;
                    Link(edge.From, edge.To);
                }
            }

            return result;
        }
    }
}
