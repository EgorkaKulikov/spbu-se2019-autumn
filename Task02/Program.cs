using System;
using System.IO;

namespace Task02
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Generating output graph...");
            GraphUtils.GenerateGraph(Config.graphPath
                , Config.numVertexes
                , Config.numEdges);
            
            var graphMatrix = GraphUtils.ReadFromFile(Config.graphPath);
            var graphSolver = new ParallelGraphSolver(graphMatrix, Config.numVertexes);

            Console.WriteLine("Running Kruskal's algorithm...");
            var kruskalSolverResult = graphSolver.ParallelKruskalSolve();
            using (StreamWriter sw = File.CreateText(Config.resultsKruskalPath))
            {
                sw.WriteLine(kruskalSolverResult);
            }
            Console.WriteLine("Finished!");

            Console.WriteLine("Running Prim's algorithm...");
            var primSolverResult = graphSolver.ParallelPrimSolve();
            using (StreamWriter sw = File.CreateText(Config.resultsPrimPath))
            {
                sw.WriteLine(primSolverResult);
            }
            Console.WriteLine("Finished!");

            if (primSolverResult == kruskalSolverResult)
            {
                Console.WriteLine("Minimal spanning tree algorithms results are matching!");
            }
            else
            {
                Console.WriteLine("Error: Minimal spanning tree algorithms results are not matching!");
            }

            Console.WriteLine("Running Floyd's algorithm...");
            var floydSolverResult = graphSolver.ParallelFloydSolve();
            Console.WriteLine("Printing output...");
            GraphUtils.PrintToFile(Config.resultsFloydPath, floydSolverResult);
            Console.WriteLine("Finished!");

            Console.ReadKey();
        }
    }
}
