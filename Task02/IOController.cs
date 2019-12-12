using System;
using System.IO;

namespace Task02
{
    class IOController
    {
        #region Exception messages
        private string FileNotFoundExceptionMessage = "File not found!";
        private string InvalidInputExceptionMessage = "Invalid input!";
        #endregion

        private string ProjectDirectory 
        { 
            get
            {
                return Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName;
            }
        }

        private (int, int, int) GenerateRandomEdge(int verticesNumber)
        {
            int from, to;
            var random = new Random();

            int cost = random.Next(Default.EdgeMinCost, Default.EdgeMaxCost);

            do
            {
                from = random.Next(0, verticesNumber);
                to   = random.Next(0, verticesNumber);
            } while (from >= to);

            return (from, to, cost);
        }

        public Graph GenerateGraph(string name, int verticesNumber = Default.VerticesNumber, int edgesNumber = Default.EdgesNumber)
        {
            int[,] matrix = new int[verticesNumber, verticesNumber];
            var maxVerticesNum = verticesNumber * (verticesNumber - 1) / 2;

            if (edgesNumber > maxVerticesNum)
                edgesNumber = maxVerticesNum;

            name = Path.ChangeExtension(name, ".txt");
            string pathToGraphFile = Path.Combine(ProjectDirectory, name);
            using (StreamWriter output = File.CreateText(pathToGraphFile))
            {
                output.WriteLine(verticesNumber);
                for (int i = 0; i < edgesNumber; i++)
                {
                    (int from, int to, int cost) edge;
                    do
                    {
                        edge = GenerateRandomEdge(verticesNumber);
                    } while (matrix[edge.from, edge.to] != 0);

                    matrix[edge.from, edge.to] = matrix[edge.to, edge.from] = edge.cost;
                    output.WriteLine($"{edge.from} {edge.to} {edge.cost}");
                }
            }

            return new Graph(matrix);
        }

        public Graph ReadGraphFrom(string path)
        {
            string pathToGraphFile = Path.Combine(ProjectDirectory, path);
            if (!Path.HasExtension(pathToGraphFile))
                pathToGraphFile += ".txt";

            if (!File.Exists(pathToGraphFile))
            {
                throw new FileNotFoundException(FileNotFoundExceptionMessage + $" ({pathToGraphFile})", pathToGraphFile);
            }

            int[,] matrix;
            int n;

            using (StreamReader input = File.OpenText(pathToGraphFile))
            {
                try
                {
                    n = Convert.ToInt32(input.ReadLine());
                    matrix = new int[n, n];
                    int edgesRead = 0;

                    while (!input.EndOfStream && edgesRead <= n * n / 2)
                    {
                        var rawEdge = input.ReadLine().Split(' ');
                        int from    = Convert.ToInt32(rawEdge[0]);
                        int to      = Convert.ToInt32(rawEdge[1]);
                        int cost    = Convert.ToInt32(rawEdge[2]);
                        matrix[from, to] = matrix[to, from] = cost;
                        edgesRead++;
                    }
                }
                catch (Exception e)
                {
                    throw new FileLoadException(InvalidInputExceptionMessage + $" ({pathToGraphFile})", pathToGraphFile, e);
                }
            }

            return new Graph(matrix);
        }

        public void CreateOutputFile(string name, string text)
        {
            name = Path.ChangeExtension(name, ".txt");
            var path = Path.Combine(ProjectDirectory, name);

            using (StreamWriter output = File.CreateText(path))
            {
                output.WriteLine(text);
            }
        }
    }
}
