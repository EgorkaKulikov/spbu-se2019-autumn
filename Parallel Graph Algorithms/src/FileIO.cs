using System.Collections.Generic;
using System.IO;

namespace GraphParallelAlgorithms
{
    public static class FileIO
    {
        public static void ReadGraph(this Graph g, string fileName)
        {
            using (var reader = new StreamReader(fileName))
            {
                g.VerticesNumber = int.Parse(reader.ReadLine());
                g.AdjacencyMatrix = new long[g.VerticesNumber, g.VerticesNumber];
                g.Edges = new List<Graph.Edge>();

                while (!reader.EndOfStream)
                {
                    var edge = reader.ReadLine().Split();

                    var start = int.Parse(edge[0]) - 1;
                    var end = int.Parse(edge[1]) - 1;
                    var weight = int.Parse(edge[2]);

                    g.Edges.Add(new Graph.Edge(start, end, weight));
                    g.AdjacencyMatrix[start, end] = weight;
                    g.AdjacencyMatrix[end, start] = weight;
                }

                g.EdgesNumber = g.Edges.Count;
            }
        }

        public static void WriteFloyd(this Graph g)
        {
            using (var writer = new StreamWriter("floyd.txt"))
            {
                var res = g.RunParallelForFloyd();

                for (var i = 0; i < g.VerticesNumber; ++i)
                {
                    for (var j = 0; j < g.VerticesNumber; ++j) writer.Write($"{res[i, j]} ");
                    writer.WriteLine();
                }
            }
        }

        public static void WriteKruskal(this Graph g)
        {
            using (var writer = new StreamWriter("kruskal.txt"))
            {
                writer.WriteLine($"{g.RunParallelForKruskal()}");
            }
        }

        public static void WritePrim(this Graph g)
        {
            using (var writer = new StreamWriter("prim.txt"))
            {
                writer.WriteLine($"{g.RunParallelForPrim()}");
            }
        }
    }
}