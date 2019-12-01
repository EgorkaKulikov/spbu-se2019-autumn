using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Task_02
{
    public static class ReadGraph
    {
        public static Dictionary<int, List<int>> Read()
        {
            var reader = new StreamReader("test.txt");
            var graph = new Dictionary<int, List<int>>();
            int.TryParse(reader.ReadLine(), out var vertex);
            for (var i = 0; i < vertex; i++)
            {
                graph[i] = new List<int>(new int[vertex]);
            }

            while (!reader.EndOfStream)
            {
                var str = reader.ReadLine();
                var info = str.Split(new char[] {' '});
                int.TryParse(info[0], out var i);
                int.TryParse(info[1], out var j);
                int.TryParse(info[2], out var weight);
                graph[i][j] = weight;
                graph[j][i] = weight;


            }

            for (var i = 0; i < vertex; i++)
            {
                for (var j = 0; j < vertex; j++)
                {
                    if (graph[i][j] == 0)
                    {
                        graph[i][j] = int.MaxValue;
                    }
                }
            }
            return graph;
        }
        
        public static List<Edge> ReadStruct()
        {
            var reader = new StreamReader("test.txt");
            var graphOfStructure = new List<Edge>();
            int.TryParse(reader.ReadLine(), out var vertex);

            while (!reader.EndOfStream)
            {
                var str = reader.ReadLine();
                var info = str.Split(new char[] {' '});
                int.TryParse(info[0], out var i);
                int.TryParse(info[1], out var j);
                int.TryParse(info[2], out var weight);
                var edge = new Edge(i, j) {Weight = weight};
                graphOfStructure.Add(edge);
            }
            
            return graphOfStructure;
        }
    }
}