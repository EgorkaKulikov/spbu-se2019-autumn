using System;
using System.Threading.Tasks;

namespace Task02
{
    public class ParallelAlgoFloyd
    {
        static int[,] dist;

        static void parallelProc(int n, int k, int i)
        {
            for (int j = 0; j < n; j++)
            {
                if (j != k)
                {
                    if (dist[i, j] > dist[i, k] + dist[k, j])
                        dist[i, j] = dist[i, k] + dist[k, j];
                }
            }
        }

        public static int[,] Execute(Graph graph)
        {
            int n = graph.graphAmountVertexes;
            dist = new int[n, n];
            Task[] tasks = new Task[n - 1];
            Array.Copy(graph.graphMatrix, dist, n * n);

            for (int k = 0; k < n; k++)
            {
                for (int i = 0; i < n; i++)
                {
                    if (i != k)
                    {
                        int newK = k;
                        int newI = i;
                        // передаём newK и newI для того, чтобы избежать замыкания
                        tasks[i < k ? i : i - 1] = Task.Run(() => parallelProc(n, newK, newI));
                    }
                }
                // ждём выполнения всех задач для того, чтобы случайно не получилось так, что
                // задача с большим k не обогнала задачу с меньшим k (и чтобы нам всегда хватило
                // ровно n - 1 задачи)
                Task.WaitAll(tasks);
            }

            return dist;
        }
    }
}
