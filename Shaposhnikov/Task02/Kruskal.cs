using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Task02
{
    public class Kruskal
    {
        private readonly int[] _parent;
        private readonly int[] _rank;
        public Kruskal(int vertices)
        {
            _parent = new int[vertices];
            _rank = new int[vertices];
        }
        public int ExecKruskal(List<Edge> edges, int vertices)
        {
            int cost = 0;
            Edge[] edgesArray = edges.ToArray();

            ParallelQs(edgesArray);
            
            for (int i = 0; i < vertices; i++)
                make_set(i);

            foreach (var edge in edgesArray)
            {
                if (find_set(edge.First) != find_set(edge.Second)) {
                    cost += edge.Weight;
                    union_sets(edge.First, edge.Second);
                }
            }

            return cost;
        }
        
        private void make_set(int v) {
            _parent[v] = v;
            _rank[v] = 0;
        }
        
        private int find_set(int v) {
            if (v == _parent[v])
                return v;
            return _parent[v] = find_set(_parent[v]);
        }

        private void union_sets(int a, int b) {
            a = find_set(a);
            b = find_set(b);
            if (a != b) {
                if (_rank[a] < _rank[b])
                    Helper.Swap(ref a, ref b);
                _parent[b] = a;
                if (_rank[a] == _rank[b])
                    _rank[a]++;
            }
        }

        private void ParallelQs<T>(T[] items) where T : IComparable<T>
        {
            ParallelQs(items, 0, items.Length);
        }

        private void ParallelQs<T>(T[] items, int left, int right)
            where T : IComparable<T>
        {
            if (right - left < 2) return;
            int pivot = Partition(items, left, right);
            if (right - left > 500)
            {
                Parallel.Invoke(
                    () => ParallelQs(items, left, pivot),
                    () => ParallelQs(items, pivot + 1, right)
                );
            }
            else
            {
                ParallelQs(items, left, pivot);
                ParallelQs(items, pivot + 1, right);
            }
        }

        private int Partition<T>(T[] items, int left, int right)
            where T : IComparable<T>
        {
            int pivotPos = (left + right) / 2;
            T pivotValue = items[pivotPos];

            Helper.Swap(ref items[right - 1], ref items[pivotPos]);
            int store = left;

            for (int i = left; i < right - 1; ++i)
            {
                if (items[i].CompareTo(pivotValue) < 0)
                {
                    Helper.Swap(ref items[i], ref items[store]);
                    ++store;
                }
            }

            Helper.Swap(ref items[right - 1], ref items[store]);
            return store;
        }
    }
}

