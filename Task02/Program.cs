using System;
using System.IO;

namespace Task02
{
    class Program
    {
        static void Main(string[] args)
        {
            StreamReader sr = new StreamReader("graph.txt");
            int size = Int32.Parse(sr.ReadLine());
            int[,] graph = new int[size, size];

            for (int i = 0; i < size; i++)
                for (int j = 0; j < size; j++)
                    if (i != j)
                        graph[i, j] = Int32.MaxValue;

            while (!sr.EndOfStream)
            {
                var edge = sr.ReadLine().Split();
                graph[Int32.Parse(edge[0]), Int32.Parse(edge[1])] = Int32.Parse(edge[2]);
                graph[Int32.Parse(edge[1]), Int32.Parse(edge[0])] = Int32.Parse(edge[2]);
            }
            var floyd = new Floyd();
            var prim = new Prim();
            var kruskal = new Kruskal();
            floyd.printResult(graph, size);
            prim.printResult(graph, size);
            kruskal.printResult(graph, size);
        }
    }
}
