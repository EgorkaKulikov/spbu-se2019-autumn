using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Task_02
{
    //generate file with graph
    internal class BuildGraph
    {
        internal static void Build()
        {
            Console.WriteLine("Write number of vertex");
            var check1 = int.TryParse(Console.ReadLine(), out var vertex);
            Console.WriteLine("Write number of edges");
            var check2 = int.TryParse(Console.ReadLine(), out var edges);
            
            if (!(check1 && check2))
            {
                Console.WriteLine("Incorrect data");
                return;
            }
            
            if (vertex < 500
                || edges < 100000
                || edges > (vertex * vertex - vertex) / 2)
            {
                Console.WriteLine("Incorrect data");
                return;
            }
            
            var writer = new StreamWriter("graph.txt");
            var generateOfNumber = new Random();
            var graph = new Dictionary<int, List<int>>();
            writer.WriteLine(vertex);
            for (var i = 0; i < vertex; i++)
            {
                graph[i] = new List<int>(new int[vertex]);
            }

            var countOfEdges = 0;
            while (countOfEdges != edges)
            {
                var i = generateOfNumber.Next(0, vertex);
                var j = generateOfNumber.Next(i, vertex);
                if (i >= j || graph[i][j] != 0) continue;
                var weight = generateOfNumber.Next(100);
                graph[i][j] = weight;

                countOfEdges++;

            }

            foreach (var key in graph.Keys)
            {
                for (var i = 0; i < vertex; i++)
                {
                    if (graph[key][i] != 0)
                    {
                        writer.WriteLine($"{key} {i} {graph[key][i]}");
                    }
                }
            }
            writer.Close();
        }
    }

}
