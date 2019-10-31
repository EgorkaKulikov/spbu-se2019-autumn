using System;
using System.Threading;
using System.IO;
using System.Collections.Generic;

namespace Kruskal
{
    public class Graph
    {
        private int N = 0;
        private List<int[]> graph = new List<int[]>();
        private int[] subtreeId;
        private int weightMST = 0;

        public void Initialization()
        {
            if (Console.IsInputRedirected)
            {
                N = Convert.ToInt32(Console.ReadLine());
            }

            string s;
            string[] edges = new string[3];
            int i, j, w;
            if (Console.IsInputRedirected)
            {
                s = Console.ReadLine();
                while (s != null)
                {
                    edges = s.Split(' ');
                    i = Convert.ToInt32(edges[0]);
                    j = Convert.ToInt32(edges[1]);
                    w = Convert.ToInt32(edges[2]);
                    if (graph.Count == 0) graph.Add(new int[] {i, j, w});
                    else
                    {
                        int k = 0;
                        while (graph[k][2] < w && k < graph.Count - 1) k++;
                        if (graph[graph.Count - 1][2] < w) graph.Add(new int[] {i, j, w});
                        else graph.Insert(k, new int[] {i, j, w});
                    }

                    s = Console.ReadLine();
                }
            }
        }
        public void Kruskal()
        {
            subtreeId = new int[N];
            for (int i = 0; i < N; i++) subtreeId[i] = i;
            for (int k = 0; k < graph.Count; k++)
            {
                Thread th = new Thread(new ParameterizedThreadStart(OneThreadKruskal));
                th.Start(k);
                th.Join();

            }

            using (StreamWriter file = new StreamWriter("output"))
            {
                file.Write(weightMST);
            }
        }

        private void OneThreadKruskal(Object ver)
        {
            int united = 0;
            int k = (int) ver;
            if (united != N)
            {
                united = 0;
                int i = graph[k][0], j = graph[k][1], w = graph[k][2];
                if (subtreeId[i] != subtreeId[j])
                {
                    weightMST += w;
                    for (int l = 0; l < N; l++)
                    {
                        if (subtreeId[l] == subtreeId[j]) united++;
                        else if (subtreeId[l] == subtreeId[i])
                        {
                            united++;
                            subtreeId[l] = subtreeId[j];
                        }
                    }
                }
            }
        }
    }

    internal class Program
    {
        public static void Main(string[] args)
        {
            Graph graph = new Graph();
            graph.Initialization();
            graph.Kruskal();
        }
    }
}