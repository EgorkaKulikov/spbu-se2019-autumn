using System;
using System.Threading.Tasks;
using System.IO;
using System.Collections.Generic;
using System.Threading;

namespace Kruskal
{
    public class Graph
    {
        private int vertexCount = 0;
        private List<Edge> edge = new List<Edge>();
        private int[] subtreeId;
        private int weightMST = 0;
        struct Edge
        {
            public int vertex1;
            public int vertex2;
            public int weight;
            public Edge(int vertex1, int vertex2, int weight)
            {
                this.vertex1 = vertex1;
                this.vertex2 = vertex2;
                this.weight = weight;
            }
        }

        public Graph()
        {
            if (Console.IsInputRedirected)
            {
                vertexCount = Convert.ToInt32(Console.ReadLine());
            }
            string s;
            string[] fieldsOfEdge = new string[3];
            int i, j, w;
            if (Console.IsInputRedirected)
            {
                s = Console.ReadLine();
                while (s != null)
                {
                    fieldsOfEdge = s.Split(' ');
                    i = Convert.ToInt32(fieldsOfEdge[0]);
                    j = Convert.ToInt32(fieldsOfEdge[1]);
                    w = Convert.ToInt32(fieldsOfEdge[2]);
                    if (edge.Count == 0) edge.Add(new Edge(i, j, w));
                    else
                    {
                        int k = 0;
                        while (edge[k].weight < w && k < edge.Count - 1) k++;
                        if (edge[edge.Count - 1].weight < w) edge.Add(new Edge(i, j, w));
                        else edge.Insert(k, new Edge(i, j, w));
                    }
                    s = Console.ReadLine();
                }
            }
        }
        public void Kruskal()
        {
            subtreeId = new int[vertexCount];
            for (int i = 0; i < vertexCount; i++) subtreeId[i] = i;
            Task task = Task.Run(() =>
            {
                for (int k = 0; k < edge.Count; k++)
                {
                    int united = 0;
                    if (united != vertexCount)
                    {
                        united = 0;
                        int i = edge[k].vertex1, j = edge[k].vertex2, w = edge[k].weight;
                        if (subtreeId[i] != subtreeId[j])
                        {
                            weightMST += w;
                            for (int l = 0; l < vertexCount; l++)
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
            });
            task.Wait();
        }
        public void Output()
        {
            using (StreamWriter file = new StreamWriter("output"))
            {
                file.Write(weightMST);
            }
        }
    }

    internal class Program
    {
        public static void Main(string[] args)
        {
            Graph graph = new Graph();
            graph.Kruskal();
            graph.Output();
        }
    }
}