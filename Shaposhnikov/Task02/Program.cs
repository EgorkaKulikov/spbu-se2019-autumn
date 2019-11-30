using System;
using System.Collections.Generic;
using System.IO;

namespace Task02
{
    class Program
    {
        static void Main(string[] args)
        {
            string directory = Directory.GetParent(Environment.CurrentDirectory).Parent?.Parent?.FullName;
            string inputFile = Path.Join(directory, It.InputFile);

            if (!File.Exists(inputFile))
            {
                Console.WriteLine("Couldn't open input file");
                return;
            }
            
            using StreamWriter writerInput = File.CreateText(inputFile);
            Helper.GenerateGraph(writerInput);
            writerInput.Close();
            
            try
            {
                List<Edge> edges = new List<Edge>();
                int vertices = new int();
                int edgesNum = new int();
               
                //reads input file, changes [vertices] and [edgesNum] numbers and fills [edges] List
                Helper.ReadGraph(inputFile, edges, ref vertices, ref edgesNum);

                using StreamWriter writerFloyd = File.CreateText(Path.Join(directory, It.OutputFloyd));
                using StreamWriter writerKruskal = File.CreateText(Path.Join(directory, It.OutputKruskal));
                using StreamWriter writerPrim = File.CreateText(Path.Join(directory, It.OutputPrim));

                //static method
                int[,] distFloyd = Floyd.ExecFloyd(edges, vertices);
                Helper.PrintMatrix(distFloyd, writerFloyd, vertices);

                Kruskal krskl = new Kruskal(vertices);
                int costKruskal = krskl.ExecKruskal(edges, vertices);
                writerKruskal.WriteLine("{0} ", costKruskal);

                //static method
                int costPrim = Prim.ExecPrim(edges, vertices);
                writerPrim.WriteLine("{0} ", costPrim);
            }
            catch (NullReferenceException)
            {
                Console.WriteLine("Wrong data, check for file's content");
            }
            catch (IndexOutOfRangeException)
            {
                Console.WriteLine("Wrong data, index went beyond borders");
            }
        }
    }
}