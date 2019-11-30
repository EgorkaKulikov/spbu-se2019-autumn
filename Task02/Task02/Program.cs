using System;
using System.IO;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Threading;

namespace Task02
{
    class Program
    {
        static int numberOfVertex, numberOfEdges;
        static int[,] graph;
        static int[,] edges;
        static int[] min;
        static int ans;
        const int Max = 214748364;  //used as infinity

        static void Main(string[] args)
        {
            using (StreamReader sr = new StreamReader("test.txt", System.Text.Encoding.Default))
            {
                string line;
                int first, second, value, i;
                i = 0;

                if ((line = sr.ReadLine()) != null)
                    numberOfVertex = int.Parse(line);

                graph = new int[numberOfVertex, numberOfVertex];
                edges = new int[(numberOfVertex * numberOfVertex) / 2, 3];

                while ((line = sr.ReadLine()) != null)
                {
                    var sublines = line.Split(' ');
                    first = int.Parse(sublines[0]);
                    second = int.Parse(sublines[1]);
                    value = int.Parse(sublines[2]);
                    graph[first - 1, second - 1] = value;
                    graph[second - 1, first - 1] = value;
                    edges[i, 0] = first;
                    edges[i, 1] = second;
                    edges[i, 2] = value;
                    i++;
                }
                numberOfEdges = i;
            }

            ThreadPool_Prim();
            Thread_Kruskal();
            Task_Floyd();

            Console.Read();
        }

        static async void Task_Floyd()
        {
            var tasks = new List<Task>();

            for (int i = 0; i < numberOfVertex; i++)
                tasks.Add(Initiate(i));

            await Task.WhenAll(tasks.ToArray());

            for (int k = 0; k < numberOfVertex; k++)
                for (int i = 0; i < numberOfVertex; i++)
                    tasks.Add(Iteration(k, i));

            await Task.WhenAll(tasks.ToArray());

            using (StreamWriter sw = new StreamWriter("outputFloyd.txt", false, System.Text.Encoding.Default))
            {
                for (int i = 0; i < numberOfVertex; i++)
                {
                    for (int j = 0; j < numberOfVertex; j++)
                    {
                        sw.Write(graph[i, j]);
                        sw.Write(" ");
                    }
                    sw.WriteLine();
                }
            }
        }

        static Task Initiate(int i)
        {
            return Task.Run(() =>
            {
                for (int j = 0; i < numberOfVertex; i++)
                {
                    if (graph[i, j] == 0)
                        graph[i, j] = Max;
                    if (i == j)
                        graph[i, j] = 0;
                }
            });
        }

        static Task Iteration(int k, int i)
        {
            return Task.Run(() =>
            {
                for (int j = 0; j < numberOfVertex; j++)
                    if (graph[i, k] + graph[k, j] < graph[i, j] || graph[i, j] == 0)
                        graph[i, j] = graph[i, k] + graph[k, j];
            });
        }

        static void Thread_Kruskal()
        {
            ans = 0;
            int threadNum = Environment.ProcessorCount;
            Thread[] threads = new Thread[threadNum];
            int chunkSize = (numberOfVertex + 1) / threadNum;
            int[] componentNum = new int[numberOfVertex + 1];

            quicksort(edges, 0, numberOfEdges);

            for (int i = 1; i < numberOfVertex + 1; i++)
                componentNum[i] = i;

            for (int i = 0; i < numberOfEdges; i++)
            {
                int start = edges[i, 0];
                int end = edges[i, 1];
                int weight = edges[i, 2];

                if (componentNum[start] != componentNum[end])
                {
                    ans += weight;
                    int a = componentNum[start];
                    int b = componentNum[end];

                    for (int j = 0; j < threadNum; j++)
                    {
                        int chunkStart = chunkSize * j;
                        int chunkEnd = chunkSize * (j + 1);
                        if (numberOfVertex - chunkStart < 2 * chunkSize)
                            chunkEnd = numberOfVertex + 1;

                        threads[j] = new Thread(() =>
                        {
                            for (int k = chunkStart; k < chunkEnd; k++)
                            {
                                if (componentNum[k] == b)
                                    componentNum[k] = a;
                            }
                        });
                        threads[j].Start();
                    }

                    for (int j = 0; j < threadNum; j++)
                        threads[j].Join();
                }

            }

            using (StreamWriter sw = new StreamWriter("outputKruskal.txt", false, System.Text.Encoding.Default))
                sw.WriteLine(ans);
        }

        static void quicksort(int[,] array, int start, int end)
        {
            if (start >= end)
            {
                return;
            }
            int pivot = partition(array, start, end);
            quicksort(array, start, pivot - 1);
            quicksort(array, pivot + 1, end);
        }

        static int partition(int[,] array, int start, int end)
        {
            int temp;
            int marker = start;
            for (int i = start; i < end; i++)
            {
                if (array[i, 2] < array[end, 2])
                {
                    temp = array[marker, 2];
                    array[marker, 2] = array[i, 2];
                    array[i, 2] = temp;
                    temp = array[marker, 0];
                    array[marker, 0] = array[i, 0];
                    array[i, 0] = temp;
                    temp = array[marker, 1];
                    array[marker, 1] = array[i, 1];
                    array[i, 1] = temp;
                    marker += 1;
                }
            }
            temp = array[marker, 2];
            array[marker, 2] = array[end, 2];
            array[end, 2] = temp;
            temp = array[marker, 0];
            array[marker, 0] = array[end, 0];
            array[end, 0] = temp;
            temp = array[marker, 1];
            array[marker, 1] = array[end, 1];
            array[end, 1] = temp;
            return marker;
        }

        static void ThreadPool_Prim()
        {
            ans = 0;
            ThreadPool.SetMaxThreads(numberOfVertex, numberOfVertex);
            min = new int[numberOfVertex + 2];

            for (int i = 0; i < numberOfVertex; i++)
                ThreadPool.QueueUserWorkItem(FindMin, i);

            while(min[numberOfVertex + 1] != numberOfVertex)
                Thread.Sleep(0);

            using (StreamWriter sw = new StreamWriter("outputPrim.txt", false, System.Text.Encoding.Default))
                sw.WriteLine(ans);
        }

        static void FindMin(Object stateInfo)
        {
            int i = (int) stateInfo;
            min[Thread.CurrentThread.ManagedThreadId] = int.MaxValue;

            for (int j = 0; j < numberOfVertex; j++)
            {
                if (graph[i, j] > 0 && (graph[i, j] < min[Thread.CurrentThread.ManagedThreadId]))
                    min[Thread.CurrentThread.ManagedThreadId] = graph[i, j];
            }

            ans += min[Thread.CurrentThread.ManagedThreadId];
            Interlocked.Increment(ref min[numberOfVertex + 1]);
        }
    }
}