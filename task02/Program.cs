using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Threading;

namespace Task_02
{
    internal class Program
    {
        private const string ResDirectoryPath = "C:\\Users\\gleb3\\RiderProjects\\Task_02\\Task_02\\Result";
        public static void Main()
        {
            //BuildGraph.Build();
            
            var graph = ReadGraph.Read();
            var vertex = graph.Count;
            var structGraph = ReadGraph.ReadStruct();

            Console.WriteLine(KruskalAlgo.KruskalFindMstConsistent(structGraph, vertex)
                              == KruskalAlgo.KruskalFindMstParallel(structGraph, vertex)
                ? "Consistent Kruskal's algo is equal to parallel"
                : "Incorrect, Kruskal's algo is not equal to parallel");

            Console.WriteLine(PrimaAlgo.ParallelPrim(graph, vertex)
                              == PrimaAlgo.SeqPrim2(graph)
                ? "Consistent Prima's algo is equal to parallel"
                : "Incorrect, Prima's algo is not equal to parallel");

            var graphOfConsistentFloyd = FloydAlgo.BuildFloydConsistent(graph);
            var graphOfParallelFloyd = FloydAlgo.BuildFloydCParallel(graph);
            Console.WriteLine(graphOfConsistentFloyd == graphOfParallelFloyd
                ? "Consistent Floyd's algo is equal to parallel"
                : "Incorrect, Floyd's algo is not equal to parallel");
            
            Directory.CreateDirectory(ResDirectoryPath);
            var path = Path.Combine(ResDirectoryPath, "result");
            var file = new StreamWriter(path);
            
            file.WriteLine($"Kruskal: {KruskalAlgo.KruskalFindMstParallel(structGraph, vertex)}");
            file.WriteLine($"Prima: {PrimaAlgo.ParallelPrim(graph, vertex)}");
            file.WriteLine("Floyd:");
            foreach (var key in FloydAlgo.BuildFloydCParallel(graph))
            {
                foreach (var value in key.Value)
                {
                    file.Write($"{value} ");
                }
                Console.WriteLine();
            }
            file.Close();
        }

    }
}