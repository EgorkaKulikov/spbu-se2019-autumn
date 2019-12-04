using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Task02
{
    static class Program
    {
        static void Main(string[] args)
        {
            using StreamReader sr = File.OpenText("input.txt");
            var firstLine = sr.ReadLine();
            if (firstLine == null)
            {
                Console.WriteLine("Error! No input given");
                return;
            }

            var verticesCount = int.Parse(firstLine);
            var adjMatrix = new int[verticesCount][];
            var edges = new List<Tuple<int, int, int>>();

            for (int i = 0; i < verticesCount; i++)
            {
                adjMatrix[i] = new int[verticesCount];

                for (int j = 0; j < verticesCount; j++)
                {
                    if (i != j) adjMatrix[i][j] = GraphAlgorithms.maxValue;
                    else adjMatrix[i][j] = 0;
                }
            }

            string str;
            while ((str = sr.ReadLine()) != null)
            {
                int[] s = str.Split(' ').ToList().ConvertAll(int.Parse).ToArray();
                adjMatrix[s[0]][s[1]] = s[2];
                adjMatrix[s[1]][s[0]] = s[2];
                edges.Add(Tuple.Create(s[0], s[1], s[2]));
            }
            
            Console.WriteLine("Input finished");

            var graphAlgorithms = new GraphAlgorithms();

            int[][] result = graphAlgorithms.Floyd(adjMatrix);
            
            Console.WriteLine("Floyd finished");

            using (StreamWriter sw = new StreamWriter("FloydOutput.txt"))
            {
                for (int i = 0; i < verticesCount; i++)
                {
                    for (int j = 0; j <= i; j++)
                    {
                        if (result[i][j] == GraphAlgorithms.maxValue)
                        {
                            result[i][j] = -1;
                        }
                        sw.Write(result[i][j]);
                        sw.Write(' ');
                    }

                    sw.Write('\n');
                }
            }
            
            Console.WriteLine("Floyd output finished");

            using (StreamWriter sw = new StreamWriter("PrimOutput.txt"))
            {
                sw.WriteLine(graphAlgorithms.Prim(adjMatrix));
            }

            using (StreamWriter sw = new StreamWriter("KruskalOutput.txt"))
            {
                sw.WriteLine(graphAlgorithms.Kruskal(edges.ToArray(), verticesCount));
            }
        }
    }
}