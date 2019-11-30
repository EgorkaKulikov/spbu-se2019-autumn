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
        public static void Main(string[] args)
        {
            //BuildGraph.Build();
            var graph = ReadGraph.Read();
            var vertex = graph.Count;
            var structGraph = ReadGraph.ReadStruct();
            /*if (KruskalAlgo.KruskalFindMstConsistent(structGraph, vertex) 
                == KruskalAlgo.KruskalFindMstParallel(structGraph, vertex))
            {
                Console.WriteLine("Consistent Kruskal's algo is equal to parallel");
            }
            else Console.WriteLine("Incorrect, Kruskal's algo is not equal to parallel");*/
            Console.WriteLine(KruskalAlgo.KruskalFindMstParallel(structGraph, vertex));
            Console.WriteLine(PrimaAlgo.PrimFindMstConsistent(structGraph, vertex));

            var graphOfConsistentFloyd = FloydAlgo.BuildFloydConsistent(graph);
            var graphOfParallelFloyd = FloydAlgo.BuildFloydCParallel(graph);
            if (graphOfConsistentFloyd == graphOfParallelFloyd)
            {
                Console.WriteLine("Consistent Floyd's algo is equal to parallel");
            }
            else Console.WriteLine("Incorrect, Prima's algo is not equal to parallel");
            Console.WriteLine();
            Directory.CreateDirectory(ResDirectoryPath);

        }

        private static void PrintPrim(Dictionary<int, List<int>> graph)
        {
            
        }

        private static void PrintKruskal()
        {
            
        }

        private static void PrintFloyd()
        {
            
        }
    }
}