using System;
using System.Threading;
using System.Collections.Generic;

namespace Task02
{
    public class ParallelAlgoPrim
    {
        static int minDistVertex; // вершина, не лежащая в дереве, расстояние до которой от дерева минимально
        static int newMinDistVertex;
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
            int localMinDistVertex = -1; // локальный ответ

            for (int to = LIndex; to < RIndex; to++)
            {
                if (!tree.Contains(to))
                {
                    // обновление минимального расстояния после добавления v в дерево
                    if (minDistToTree[to] > graph.graphMatrix[minDistVertex, to])
                    {
                        minDistToTree[to] = graph.graphMatrix[minDistVertex, to];
                    }

                    // обновление локального ответа
                    if (localMinDistVertex == -1 || minDistToTree[localMinDistVertex] > minDistToTree[to])
                        localMinDistVertex = to;
                }
            }

            if (localMinDistVertex != -1)
            {
                mutNewV.WaitOne();
                // обновление глобального ответа
                if (newMinDistVertex == -1 || minDistToTree[newMinDistVertex] > minDistToTree[localMinDistVertex])
                    newMinDistVertex = localMinDistVertex;
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
            minDistVertex = 0;

            while (minDistVertex != -1)
            {
                ans += minDistToTree[minDistVertex];
                tree.Add(minDistVertex);

                newMinDistVertex = -1;

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

                minDistVertex = newMinDistVertex;
            }

            return ans;
        }
    }
}
