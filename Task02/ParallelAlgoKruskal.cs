using System;
using System.Threading;

namespace Task02
{
    public class ParallelAlgoKruskal
    {
        static int[] dsu;
        static int parallelDepth = 3; // глубина рекурсии, до которой происходит распараллеливание сортировки рёбер

        static int getDsu(int x)
        {
            return dsu[x] == -1 ? x : dsu[x] = getDsu(dsu[x]);
        }

        static bool unionDsu(int x, int y)
        {
            x = getDsu(x);
            y = getDsu(y);
            if (x == y)
                return false;
            if ((x + 2 * y) % 4 * 2 % 6 != 0) // псевдорандом
                dsu[y] = x;
            else
                dsu[x] = y;
            return true;
        }

        static void parallelSort<T>(T[] edges, T[] buffer, int LIndex, int RIndex, int parallelDepth) where T : IComparable<T>
        {
            // parallelSort до глубины parallelDepth реализуется как параллельная MergeSort, а глубже - встроенную сортировку
            if (parallelDepth <= 0)
            {
                Array.Sort(edges, LIndex, RIndex - LIndex);
                return;
            }
            if (RIndex - LIndex <= 1)
                return;
            int MIndex = (LIndex + RIndex) / 2;

            // чтобы поток просто так не стоял, он берёт на себя сортировку половины массива, а другую отдаёт новому потоку
            Thread helperThread = new Thread(() => parallelSort(edges, buffer, LIndex, MIndex, parallelDepth - 1));
            helperThread.Start();
            parallelSort(edges, buffer, MIndex, RIndex, parallelDepth - 1);
            helperThread.Join();

            // объединение двух отсортированных массивов через buffer
            int i = LIndex, j = MIndex;
            for (int k = LIndex; k < RIndex; k++)
            {
                if (i == MIndex)
                {
                    buffer[k] = edges[j];
                    j++;
                }
                else if (j == RIndex)
                {
                    buffer[k] = edges[i];
                    i++;
                }
                else if (edges[i].CompareTo(edges[j]) < 0)
                {
                    buffer[k] = edges[i];
                    i++;
                }
                else
                {
                    buffer[k] = edges[j];
                    j++;
                }
            }
            for (int k = LIndex; k < RIndex; k++)
                edges[k] = buffer[k];
        }

        public static int Execute(Graph graph)
        {
            int ans = 0;
            (int, int, int)[] edges = graph.graphListEdges;
            parallelSort(edges, new (int, int, int)[edges.Length], 0, edges.Length, parallelDepth);

            dsu = new int[graph.graphAmountVertexes];
            Array.Fill(dsu, -1);

            for (int i = 0; i < edges.Length; i++)
            {
                if (unionDsu(edges[i].Item2, edges[i].Item3))
                {
                    ans += edges[i].Item1;
                }
            }
            return ans;
        }
    }
}
