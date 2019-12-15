using System;
using System.Diagnostics;

namespace Task02
{
    class Program
    {
        static void Main()
        {
            Graph.GenerateGraph(AppDomain.CurrentDomain.BaseDirectory + "/2.in",
                Constants.VerticesNumber,
                Constants.VerticesNumber * (Constants.VerticesNumber - 1) / 2,
                100);

            var edgesArray = Graph.LoadArrayGraph(AppDomain.CurrentDomain.BaseDirectory + "/2.in");

            //create distinct arrays to make fair estimations
            var edgesArray1 = new Graph.Edge[edgesArray.Length];
            Array.Copy(edgesArray, edgesArray1, edgesArray.Length);
            var edgesArray2 = new Graph.Edge[edgesArray.Length];
            Array.Copy(edgesArray, edgesArray2, edgesArray.Length);

            Console.WriteLine("generated");
            Stopwatch timer1 = Stopwatch.StartNew();
            var a = Kruskal.RunSeqKruskal(Constants.VerticesNumber, edgesArray1);
            // var a = Floyd.RunSeqFloyd(graphMatrix);
            // var a = Prim.RunSeqPrim(graphMatrix);
            timer1.Stop();
            Console.WriteLine("Прошло: {0} мс.",
                timer1.ElapsedMilliseconds);

            Stopwatch timer2 = Stopwatch.StartNew();
            var b = Kruskal.RunParallelKruskal(Constants.VerticesNumber, edgesArray2);
            // var b = Floyd.RunSeqFloyd(graphMatrix);
            // var b = Prim.RunParallelPrim(graphMatrix);
            timer2.Stop();
            Console.WriteLine("Прошло: {0} мс.",
                timer2.ElapsedMilliseconds);
            
            //validate results
            for (int i = 0; i < a.Length; i++)
            {
                if (a[i].weight != b[i].weight) Console.WriteLine(a[i].weight == b[i].weight);
            }
        }
    }
}