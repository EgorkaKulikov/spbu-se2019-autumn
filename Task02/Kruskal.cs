using System;
using System.IO;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Task02
{
    class Kruskal
    {
        public class Edge
        {
            public int start;
            public int end;
            public int weight;

            public Edge(int _start, int _end, int _weight)
            {
                start = _start;
                end = _end;
                weight = _weight;
            }
        }

        public List<Edge> getEdges(int[,] graph, int size)
        {
            List<Edge> edges = new List<Edge>();

            for (int i = 0; i < size; i++)
            {
                for (int j = i + 1; j < size; j++)
                {
                    if (graph[i, j] != Int32.MaxValue)
                        edges.Add(new Edge(i, j, graph[i, j]));
                }
            }
            return edges;
        }

        public int kruskalAlg(int[,] graph, int size)
        {
            var edges = getEdges(graph, size);
            int weight = 0;
            quickSort(edges, 0, edges.Count-1);
            int[] treeId = new int[size];
            for (int i = 0; i < size; i++)
                treeId[i] = i;
            for (int i = 0; i < edges.Count; i++)
            {
                int start = edges[i].start, end = edges[i].end;
                if (treeId[start] != treeId[end])
                {
                    weight += edges[i].weight;
                    int oldId = treeId[end], newId = treeId[start];
                    for (int j = 0; j < size; j++)
                        if (treeId[j] == oldId)
                            treeId[j] = newId;
                }
            }
            return weight;
        }

        public void quickSort(List<Edge> edges, int left, int right)
        {
            if (left < right)
            {
                int leftptr = left;
                int rightptr = right;
                var middle = edges[(leftptr + rightptr) / 2];
                do
                {
                    while (middle.weight > edges[leftptr].weight) leftptr++;
                    while (edges[rightptr].weight > middle.weight) rightptr--;
                    if (leftptr <= rightptr)
                    {
                        var tmp = edges[leftptr];
                        edges[leftptr] = edges[rightptr];
                        edges[rightptr] = tmp;
                        leftptr++;
                        rightptr--;
                    }
                } while (leftptr <= rightptr);
                
                Task leftSort = Task.Run(() => quickSort(edges, left, rightptr));
                Task rightSort = Task.Run(() => quickSort(edges, leftptr, right));
                Task.WaitAll(leftSort, rightSort);
            }
        }


        public void printResult(int[,] graph, int size)
        {
            using (var sw = new StreamWriter("Kruskal.txt"))
            {
                var result = kruskalAlg(graph, size);
                sw.Write(result);
            }
        }
    }
} 