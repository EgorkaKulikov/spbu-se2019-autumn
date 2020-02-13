using System;
using System.IO;

namespace Task02
{
    class Program
    {
        static void Main(string[] args)
        {
            var graph = new Graph();
            graph.ReadGraph("input.txt");

            var seqFloydResult = Algorithms.Floyd.SeqSolve(graph.adjacencyMatrix, 100);
            var parFloydResult = Algorithms.Floyd.ParSolve(graph.adjacencyMatrix, 100);

            int seqPrimResult = Algorithms.Prim.SeqSolve(graph.adjacencyMatrix, graph.vertexCnt);
            int parPrimResult = Algorithms.Prim.ParSolve(graph.adjacencyMatrix, graph.vertexCnt);

            int seqKruskalResult = Algorithms.Kruskal.SeqSolve(graph.edges.ToArray(), graph.vertexCnt);
            int parKruskalResult = Algorithms.Kruskal.ParSolve(graph.edges.ToArray(), graph.vertexCnt);

            using var writer = File.CreateText("result.txt");
            for (int i = 0; i < graph.vertexCnt; ++i)
                for (int j = 0; j < graph.vertexCnt; ++j)
                    if (seqFloydResult[i, j] != parFloydResult[i, j])
                        writer.WriteLine($"Error in [{i}, {j}]!");

            writer.WriteLine($"Sequential Kruskal: {seqKruskalResult}\n" +
                $"Parallel Kruskal: {parKruskalResult}\n" +
                $"Sequential Prim: {seqPrimResult}\n" +
                $"Parallel Prim: {parPrimResult}");
        }
    }
}
