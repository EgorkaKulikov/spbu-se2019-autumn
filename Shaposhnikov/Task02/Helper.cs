using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Task02
{
    public static class Helper
    {
        public static readonly Random Rand = new Random();
        
        public static void GenerateGraph(StreamWriter writer)
        {
            int vertices = Rand.Next(It.MinVerticesNum, It.MaxVerticesNum + 1);
            int edges = Rand.Next(It.MinEdgesNum, vertices * (vertices - 1) / 2 + 1);
            int[,] graph = new int[vertices, vertices];
            int[] visited = Enumerable.Repeat(0, vertices).ToArray();
            writer.WriteLine("{0} {1}", vertices, edges);
            
            for (int i = 0; i < edges; i++)
            {
                int start, end;
                do
                {
                    start = Rand.Next(0, vertices - 1);
                    end = Rand.Next(start + 1, vertices);
                } while (visited[start] == vertices - 1 || graph[start, end] != 0);
                visited[end]++;
                visited[start]++;
                graph[start, end] = graph[end, start] = 1;
                int weight = Rand.Next(It.MinEdgeWeight, It.MaxEdgeWeight);
                writer.WriteLine("{0} {1} {2}", start, end, weight);
            } 
        }

        public static int[,] GetIncidenceMatrix(List<Edge> edges, int vertices)
        {
            int[,] dist = new int[vertices, vertices];
            for (int i = 0; i < vertices; i++)
            {
                for (int j = 0; j < vertices; j++)
                {
                    dist[i, j] = It.Inf;
                }
            }
            
            for (int i = 0; i < vertices; i++)
                dist[i, i] = 0;
            
            foreach (var edge in edges)
            {
                int start = edge.First, end = edge.Second, weight = edge.Weight;
                dist[start, end] = dist[end, start] = weight;
            }

            return dist;
        }

        public static void PrintMatrix(int[,] matrix, StreamWriter writer, int vertices)
        {
            for (int i = 0; i < vertices; i++)
            {
                for (int j = 0; j < vertices; j++)
                {
                    writer.Write("{0} ", matrix[i, j]);
                }
                writer.WriteLine();
            }
        }
        
        public static void Swap<T>(ref T a, ref T b)
        {
            T temp = a;
            a = b;
            b = temp;
        }

        public static void ReadGraph(string inputFile, List<Edge> edges, ref int vertices, ref int edgesNum)
        {
            using StreamReader reader = new StreamReader(inputFile);
            var input = reader.ReadLine().Split(' ');
            vertices = int.Parse(input[0]);
            edgesNum = int.Parse(input[1]);
            
            for (int i = 0; i < edgesNum; i++)
            {
                input = reader.ReadLine().Split(' ');
                int start = int.Parse(input[0]);
                int end = int.Parse(input[1]);
                int weight = int.Parse(input[2]);
                edges.Add(new Edge(start, end, weight));
                edges.Add(new Edge(start, end, weight));
            }
        }
    }
}