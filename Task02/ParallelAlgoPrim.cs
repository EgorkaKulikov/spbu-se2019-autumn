using System;
using System.Threading;
using System.Collections.Generic;

namespace Task02
{
    public class ParallelAlgoPrim
    {
        static int v; // вершина, не лежащая в дереве, расстояние до которой от дерева минимально
        static int newV;
        static int amountThreads = 6;
        static int runningThreads; // количество выполняющихся потоков
        static int chunkSize;
        static Mutex mutRunningThreads = new Mutex(); // блокировка при изменении runningThreads
        static Mutex mutNewV = new Mutex(); // блокировка при чтении и изменении newV
        static HashSet<int> tree; // вершины, лежащие в остовном дереве
        static int[] minDistToTree;

        static void FindNewV(Graph graph, int LIndex, int RIndex)
        {
            // нахождение очередной вершины с минимальным расстоянием до дерева
            int localV = -1; // локальный ответ

            for (int to = LIndex; to < RIndex; to++)
            {
                if (!tree.Contains(to))
                {
                    // обновление минимального расстояния после добавления v в дерево
                    if (minDistToTree[to] > graph.graphMatrix[v, to])
                    {
                        minDistToTree[to] = graph.graphMatrix[v, to];
                    }

                    // обновление локального ответа
                    if (localV == -1 || minDistToTree[localV] > minDistToTree[to])
                        localV = to;
                }
            }

            if (localV != -1)
            {
                mutNewV.WaitOne();
                // обновление глобального ответа
                if (newV == -1 || minDistToTree[newV] > minDistToTree[localV])
                    newV = localV;
                mutNewV.ReleaseMutex();
            }

            mutRunningThreads.WaitOne();
            runningThreads--;
            mutRunningThreads.ReleaseMutex();
        }

        public static int Execute(Graph graph)
        {
            int ans = 0;
            tree = new HashSet<int>();
            minDistToTree = new int[graph.graphAmountVertexes];
            Array.Fill(minDistToTree, Graph.INF);
            chunkSize = graph.graphAmountVertexes / amountThreads;

            // начинаем с вершины 0
            minDistToTree[0] = 0;
            v = 0;

            while (v != -1)
            {
                ans += minDistToTree[v];
                tree.Add(v);

                newV = -1;

                runningThreads = amountThreads;
                for (int i = 0; i < amountThreads - 1; i++)
                {
                    int LIndex = chunkSize * i;
                    int RIndex = chunkSize * (i + 1);
                    ThreadPool.QueueUserWorkItem(_ => FindNewV(graph, LIndex, RIndex));
                }
                // загрзим главный поток чтобы он просто так не ждал
                FindNewV(graph, chunkSize * (amountThreads - 1), graph.graphAmountVertexes);

                while (runningThreads > 0) {}

                v = newV;
            }

            return ans;
        }
    }
}
