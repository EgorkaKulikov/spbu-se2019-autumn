//.NET Framework 4.8

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Task02
{
    class Program
    {
        static void Main()
        {
            IOController ioController = new IOController();
            //var graph = ioController.GenerateGraph("graph", 50, 1225);
            var graph = ioController.ReadGraphFrom("graph");

            var kruskalOutput  = Kruskal.Run(graph);
            var floydRawOutput = Floyd.Run(graph);
            var primOutput = Prim.Run(graph);

            string floydOutput = "";
            for (int i = 0; i < graph.VerticesNum; i++)
            {
                for (int j = 0; j < graph.VerticesNum; j++)
                    floydOutput += $"{floydRawOutput[i, j]} ";
            
                floydOutput += "\n";
            }
            
            if (kruskalOutput != primOutput)
                throw new Exception(":(");
            
            ioController.CreateOutputFile("kruskalOutput", kruskalOutput.ToString());
            ioController.CreateOutputFile("floydOutput", floydOutput);
            ioController.CreateOutputFile("primOutput", primOutput.ToString());
        }
    }
}
