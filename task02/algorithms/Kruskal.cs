using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Task02
{
    static public class Kruskal
    {
        public static List<Edge> BuildMst(Int32 numberOfVertices, List<Edge> edges)
        {
            ParallelSort(0, edges.Count);

            var ranks = Enumerable.Range(0, numberOfVertices).Select(_ => 0).ToArray();
            var setId = Enumerable.Range(0, numberOfVertices).ToArray();
            var result = new List<Edge>();

            foreach (var edge in edges)
            {
                Int32 set1 = FindSet(edge.first);
                Int32 set2 = FindSet(edge.second);

                if (set1 != set2)
                {
                    result.Add(edge);
                    UniteSets(set1, set2);
                }
            }

            return result;

            Int32 FindSet(Int32 vertex)
            {
                if (vertex == setId[vertex])
                {
                    return vertex;
                }
                return setId[vertex] = FindSet(setId[vertex]);
            }

            void UniteSets(Int32 a, Int32 b)
            {
                a = FindSet(a);
                b = FindSet(b);
                if (a != b)
                {
                    if (ranks[a] < ranks[b])
                    {
                        ChangeRank(b, a);
                    }
                    else
                    {
                        ChangeRank(a, b);
                    }
                }

                void ChangeRank(Int32 a, Int32 b)
                {
                    setId[b] = a;
                    if (ranks[a] == ranks[b])
                    {
                        ranks[a]++;
                    }
                }
            }

            void ParallelSort(Int32 begin, Int32 end)
            {
                var pivot = Partition(begin, end);
                if (end - begin > 1000)
                {
                    var task1 = Task.Run(() => ParallelSort(begin, pivot));
                    var task2 = Task.Run(() => ParallelSort(pivot + 1, end));
                    Task.WaitAll(task1, task2);
                }
                else
                {
                    SequentialSort(begin, pivot);
                    SequentialSort(pivot + 1, end);
                }
            }

            void SequentialSort(Int32 begin, Int32 end)
            {
                if (begin == end) return;
                var pivot = Partition(begin, end);
                SequentialSort(begin, pivot);
                SequentialSort(pivot + 1, end);
            }

            Int32 Partition(Int32 begin, Int32 end)
            {
                var x = edges[begin];
                Swap(begin, end - 1);

                var current = begin;
                for (var i = begin; i < end - 1; ++i)
                {
                    if (edges[i].CompareTo(x) < 0)
                    {
                        Swap(current, i);
                        current++;
                    }
                }
                Swap(current, end - 1);

                return current;

                void Swap(Int32 i, Int32 j)
                {
                    var temp = edges[i];
                    edges[i] = edges[j];
                    edges[j] = temp;
                }
            }
        }
    }
}