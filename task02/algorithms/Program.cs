using System;
using System.Collections.Generic;
using System.Threading;

using System.Linq;

namespace Task02
{
    class Program
    {
        static void Main(String[] args)
        {
            var numberOfVertices = Int32.Parse(Console.ReadLine());
            var edges = new List<Edge>();

            for (var line = Console.ReadLine(); line != null; line = Console.ReadLine())
            {
                char[] delimeters = { ' ' };
                var split = line.Split(delimeters);

                var from = Int32.Parse(split[0]);
                var to = Int32.Parse(split[1]);
                var weight = Int32.Parse(split[2]);

                edges.Add(new Edge(from, to, weight));
            }

            switch (args[0])
            {
                case "kruskal":
                    {
                        var mst = Kruskal.BuildMst(numberOfVertices, edges);
                        var totalWeight = 0;

                        foreach (var edge in mst)
                        {
                            totalWeight += edge.weight;
                        }

                        Console.WriteLine(totalWeight);
                        break;
                    }
                case "prim":
                    {
                        var mst = Prim.BuildMst(numberOfVertices, edges);

                        var totalWeight = 0;

                        foreach (var edge in mst)
                        {
                            totalWeight += edge.weight;
                        }

                        Console.WriteLine(totalWeight);
                        break;
                    }
                case "floyd":
                    {
                        var paths = Floyd.FindShortestPaths(numberOfVertices, edges);

                        for (Int32 i = 0; i < numberOfVertices; i++)
                        {
                            for (Int32 j = i + 1; j < numberOfVertices; j++)
                            {
                                Console.WriteLine($"{i} {j} {paths[i, j]}");
                            }
                        }
                        break;
                    }
            }
        }
    }
}
