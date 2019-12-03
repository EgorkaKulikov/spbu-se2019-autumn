using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace Task02
{
    public class GraphAlgorithms
    {
        public const int maxValue = Int32.MaxValue >> 1;
        
        private int[][] AdjMatrix;

        private Tuple<int, int, int>[] Edges;

        public bool FloydVerify(int[][] adjMatrix, int[][] result)
        {
            AdjMatrix = new int[adjMatrix.Length][];
            for (var i = 0; i < adjMatrix.Length; i++)
            {
                AdjMatrix[i] = new int[adjMatrix.Length];
                for (var j = 0; j < adjMatrix.Length; j++)
                {
                    AdjMatrix[i][j] = adjMatrix[i][j];
                }
            }
            for (var k = 0; k < AdjMatrix.Length; k++)
            {
                for (var i = 0; i < AdjMatrix.Length; i++)
                {
                    for (var j = 0; j < AdjMatrix.Length; j++)
                    {
                        AdjMatrix[i][j] = Math.Min(AdjMatrix[i][j], AdjMatrix[i][k] + AdjMatrix[k][j]);
                    }
                }
            }

            for (var i = 0; i < AdjMatrix.Length; i++)
            {
                for (var j = 0; j < AdjMatrix.Length; j++)
                {
                    if (AdjMatrix[i][j] != result[i][j]) return false;
                }
            }

            return true;
        }
        public int[][] Floyd(int[][] adjMatrix)
        {
            AdjMatrix = new int[adjMatrix.Length][];
            for (var i = 0; i < adjMatrix.Length; i++)
            {
                AdjMatrix[i] = new int[adjMatrix.Length];
                for (var j = 0; j < adjMatrix.Length; j++)
                {
                    AdjMatrix[i][j] = adjMatrix[i][j];
                }
            }
            
            for (var k = 0; k < AdjMatrix.Length; k++)
            {
                var innerIterations = new List<Task>();
                
                for (var i = 0; i < AdjMatrix.Length; i++)
                {
                    if (i != k)
                    {
                        innerIterations.Add(InnerFloyd(k, i));
                    }
                }

                Task.WhenAll(innerIterations.ToArray());
            }
            
            /*var result = new int[AdjMatrix.Length][];
            for (var i = 0; i < AdjMatrix.Length; i++)
            {
                result[i] = new int[AdjMatrix.Length];
                for (var j = 0; j < AdjMatrix.Length; j++)
                {
                    result[i][j] = AdjMatrix[i][j];
                }
            }
            return result;*/
            return AdjMatrix;
        }

        private Task InnerFloyd(int k, int i)
        {
            Action action = () =>
            {
                for (var j = 0; j < AdjMatrix.Length; j++)
                {
                    if (j != k)
                    {
                        AdjMatrix[i][j] = Math.Min(AdjMatrix[i][j], AdjMatrix[i][k] + AdjMatrix[k][j]);
                    }
                }
            };
            return Task.Run(action);
        }

        public int Prim(int[][] adjMatrix)
        {
            int ans = 0;
            bool[] added = new bool[AdjMatrix.Length];
            int[] minWeight = new int[AdjMatrix.Length];
            int[] minEdgeTo = new int[AdjMatrix.Length];

            for (var i = 0; i < AdjMatrix.Length; i++)
            {
                added[i] = false;
                minWeight[i] = maxValue;
                minEdgeTo[i] = -1;
            }

            minWeight[0] = 0;

            for (var i = 0; i < AdjMatrix.Length; i++)
            {
                var v = -1;

                for (int j = 0; j < AdjMatrix.Length; j++)
                {
                    if (!added[j] && (v == -1 || minWeight[j] < minWeight[v]))
                    {
                        v = j;
                    }
                }

                if (minWeight[i] == maxValue)
                {
                    return maxValue;
                }

                added[v] = true;
                ans += minWeight[v];
                var completed = 0;
                ManualResetEvent allCompleted = new ManualResetEvent(initialState: false);

                for (var to = 0; to < adjMatrix.Length; to++)
                {
                    ThreadPool.QueueUserWorkItem(toObj =>
                    {
                        var actualTo = (int) toObj;
                        if (AdjMatrix[v][actualTo] < minWeight[actualTo])
                        {
                            minWeight[actualTo] = AdjMatrix[v][actualTo];
                            minEdgeTo[actualTo] = v;
                        }
                        if (Interlocked.Increment(ref completed) == AdjMatrix.Length)
                        {
                            allCompleted.Set();
                        }
                    }, to);
                }
                
                allCompleted.WaitOne();
            }

            return ans;
        }
        
        private void Swap(int i, int j)
        {
            var tmp = Edges[i];
            Edges[i] = Edges[j];
            Edges[j] = tmp;
        }

        private int Partition(int begin, int end) 
        {
            var j = begin;
            var separator = Edges[begin].Item3;
            for (var i = begin; i <= end; i++)
            {
                if (Edges[i].Item3.CompareTo(separator) < 0)
                {
                    j++;
                    Swap(i, j);
                }
            }
            Swap(begin, j);
            return j;
        }

        private void InsertionSort(int begin, int end)
        {
            for (var i = begin + 1; i <= end; i++)
            {

                var j = i - 1;
                var x = Edges[i];
                while (j >= 0 && Edges[j].Item3.CompareTo(x.Item3) > 0)
                {
                    Edges[j+1] = Edges[j];       
                    j--;
                }
                Edges[j+1] = x;
            }
        }

        public void QuickSort(Object limits)
        {
            var (begin, end) = (Tuple<int, int>) limits;
            if (end - begin < 500) 
            {
                InsertionSort(begin, end);
            }
            else 
            {
                var separator = Partition(begin, end);
                List<Thread> threads = new List<Thread>();
                var leftLimits = Tuple.Create(begin, separator - 1);
                var rightLimits = Tuple.Create(separator + 1, end);
                if (separator - begin > 10000)
                {
                    threads.Add(new Thread(QuickSort));
                    threads.Last().Start(leftLimits);
                }
                else
                {
                    QuickSort(leftLimits);
                }

                if (end - separator > 1000000)
                {
                    threads.Add(new Thread(QuickSort));
                    threads.Last().Start(rightLimits);
                }
                else
                {
                    QuickSort(rightLimits);
                }

                foreach (var thread in threads)
                {
                    thread.Join();
                }
            }
        }

        int dsu_get(int[] dsu, int v)
        {
            if (v != dsu[v])
            {
                dsu[v] = dsu_get(dsu, dsu[v]);
            }

            return dsu[v];
        }
        
        void dsu_unite(int[] dsu, int a, int b) {
            var repA = dsu_get (dsu, a);
            var repB = dsu_get (dsu, b);
            var random = new Random();
            if ((random.Next() & 1) == 0)
            {
                var tmp = repA;
                repA = repB;
                repB = tmp;
            }

            if (repA != repB)
            {
                dsu[repA] = repB;
            }
        }

        public int Kruskal(Tuple<int, int, int>[] edges, int verticesCount)
        {
            Edges = edges;
            QuickSort(Tuple.Create(0, edges.Length - 1));
            var dsu = new int[verticesCount];
            for (var i = 0; i < verticesCount; i++)
            {
                dsu[i] = i;
            }

            int ans = 0;
            foreach (var (from, to, weight) in edges)
            {
                if (dsu_get(dsu, from) != dsu_get(dsu, to))
                {
                    ans += weight;
                    dsu_unite(dsu, from, to);
                }
            }
            
            return ans;
        }
    }
}