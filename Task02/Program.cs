using System;
using System.IO;
using System.Diagnostics;

namespace Task02
{
    class Program
    {
        static void Main()
        {
            StreamWriter foutFloyd = new StreamWriter("floyd.txt");
            StreamWriter foutKruskal = new StreamWriter("kruskal.txt");
            StreamWriter foutPrim = new StreamWriter("prim.txt");

            Console.WriteLine("Введите количество вершин нового графа:");
            string inputN = Console.ReadLine();
            int n = int.Parse(inputN);
            Console.WriteLine("Введите примерное количество рёбер нового графа:");
            string inputM = Console.ReadLine();
            int m = int.Parse(inputM);
            Console.WriteLine("Введите два числа - диапазон весов рёбер нового графа:");
            string[] inputW = Console.ReadLine().Split(" ");
            int minWeight = int.Parse(inputW[0]);
            int maxWeight = int.Parse(inputW[1]);

            Console.WriteLine("\nСоздание нового графа...");
            GenerateMatrix.Execute(n, m, minWeight, maxWeight);
            Console.WriteLine("Новый граф успешно создан.\n");

            Stopwatch timeSequentialFloyd = Stopwatch.StartNew();
            int[,] ansSequentialFloyd = SequentialAlgoFloyd.Execute();
            timeSequentialFloyd.Stop();
            Console.WriteLine($"Время исполнения последовательного алгоритма Флойда = {timeSequentialFloyd.ElapsedMilliseconds} мс.");

            Stopwatch timeParallelFloyd = Stopwatch.StartNew();
            int[,] ansParallelFloyd = ParallelAlgoFloyd.Execute();
            timeParallelFloyd.Stop();
            Console.WriteLine($"Время исполнения параллельного алгоритма Флойда = {timeParallelFloyd.ElapsedMilliseconds} мс.");

            Stopwatch timeSequentialKruskal = Stopwatch.StartNew();
            int ansSequentialKruskal = SequentialAlgoKruskal.Execute();
            timeSequentialKruskal.Stop();
            Console.WriteLine($"Время исполнения последовательного алгоритма Краскала = {timeSequentialKruskal.ElapsedMilliseconds} мс.");

            Stopwatch timeParallelKruskal = Stopwatch.StartNew();
            int ansParallelKruskal = ParallelAlgoKruskal.Execute();
            timeParallelKruskal.Stop();
            Console.WriteLine($"Время исполнения параллельного алгоритма Краскала = {timeParallelKruskal.ElapsedMilliseconds} мс.");

            Stopwatch timeParallelPrim = Stopwatch.StartNew();
            int ansParallelPrim = ParallelAlgoPrim.Execute();
            timeParallelPrim.Stop();
            Console.WriteLine($"Время исполнения параллельного алгоритма Прима = {timeParallelPrim.ElapsedMilliseconds} мс.");

            Console.WriteLine();

            bool ansEquals = true;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (ansSequentialFloyd[i, j] != ansParallelFloyd[i, j])
                    {
                        ansEquals = false;
                        break;
                    }
                }
            }
            if (ansEquals)
            {
                Console.WriteLine("OK: Ответ параллельного алгоритма Флойда совпал с последовательным.");
            }
            else
            {
                Console.WriteLine("WA: Ответ параллельного алгоритма Флойда не совпал с последовательным.");
            }

            if (ansSequentialKruskal == ansParallelKruskal)
            {
                Console.WriteLine("OK: Ответ параллельного алгоритма Краскала совпал с последовательным.");
            }
            else
            {
                Console.WriteLine("WA: Ответ параллельного алгоритма Краскала не совпал с последовательным.");
            }

            if (ansSequentialKruskal == ansParallelPrim)
            {
                Console.WriteLine("OK: Ответ параллельного алгоритма Прима совпал с последовательным алгоритмом Краскала.");
            }
            else
            {
                Console.WriteLine("WA: Ответ параллельного алгоритма Прима не совпал с последовательным алгоритмом Краскала.");
            }

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    foutFloyd.Write($"{ansParallelFloyd[i, j]}{(j + 1 == n ? '\n' : ' ')}");
                }
            }
            foutKruskal.Write(ansParallelKruskal);
            foutPrim.Write(ansParallelPrim);
            foutFloyd.Close();
            foutKruskal.Close();
            foutPrim.Close();
        }
    }
}
