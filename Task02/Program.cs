using System;
using System.IO;

namespace Graphs
{
    class Program
    {
        static void Main()
        {
            var graphInit = new Graph();

            int[,] graph = graphInit.GenerateGraphWithEdges();

            using (var writer = new StreamWriter(Paths.PathInitGraph, false, System.Text.Encoding.Default))
            {
                graphInit.PrintGraph(graph, graphInit.Size, writer);
            }

            var resolver = new Algorithms();
            var resolverParallel = new ParallelAlgorithms();

            int kruskal = resolver.SolveKruskal(graph, graphInit.Size).Item1;
            int prim = resolver.SolvePrim(graph, graphInit.Size).Item1;
            int[,] floyd = resolver.SolveFloyd(graph, graphInit.Size);

            int kruskalPar = resolverParallel.SolveKruskal(graph, graphInit.Size).Item1;
            int primPar = resolverParallel.SolvePrim(graph, graphInit.Size).Item1;
            int[,] floydPar = resolverParallel.SolveFloyd(graph, graphInit.Size);


            using (var writer = new StreamWriter(Paths.PathResFl, false, System.Text.Encoding.Default))
            {
                WriteResultsIntoFile(graphInit.CompareMatrixGraphs(floyd, floydPar, graphInit.Size)
                                    , writer
                                    , () => graphInit.PrintGraph(floydPar, graphInit.Size, writer));
            }

            using (var writer = new StreamWriter(Paths.PathResPr, false, System.Text.Encoding.Default))
            {
                WriteResultsIntoFile(prim == primPar, writer
                                    , () => writer.WriteLine(primPar));
            }

            using (var writer = new StreamWriter(Paths.PathResKr, false, System.Text.Encoding.Default))
            {
                WriteResultsIntoFile(kruskal == kruskalPar, writer
                                    , () => writer.WriteLine(kruskalPar));
            }
        }

        static void WriteResultsIntoFile(bool condition, StreamWriter writer, Action action)
        {
            if (condition)
            {
                action();
            }
            else
            {
                writer.WriteLine(Constants.NotTheSameResultLog);
            }
        }
    }
}