using System;
using System.Collections.Generic;
using System.IO;

namespace Task02
{
    public static class GraphUtils
    {
        private static readonly Random rand = new Random();

        public static void GenerateGraph(string path, int numVertexes, int numEdges)
        {
            var graphEdges = new List<Edge>();

            for (int i = 0; i < numEdges; i++)
            {
                int startVertex, endVertex;
                int weight = rand.Next(Config.maxWeight);
                Edge newEdge, newEdgeBack;

                do
                {
                    startVertex = rand.Next(numVertexes - 1);
                    endVertex = rand.Next(startVertex + 1, numVertexes);
                        
                    newEdge = new Edge(startVertex, endVertex, weight);
                    newEdgeBack = new Edge(endVertex, startVertex, weight);
                } while (graphEdges.Contains(newEdge));

                graphEdges.Add(newEdge);
                graphEdges.Add(newEdgeBack);
            }

            PrintToFile(path, numVertexes, graphEdges);
        }

        public static void PrintToFile(string path, int numVertexes, List<Edge> graphEdges)
        {
            using (StreamWriter sw = File.CreateText(path))
            {
                sw.WriteLine("{0}", numVertexes);

                foreach (var edge in graphEdges)
                {
                    sw.WriteLine("{0} {1} {2}"
                        , edge.startVertex
                        , edge.endVertex
                        , edge.weight);
                }
            }
        }

        public static void PrintToFile(string path, int[,] graphMatrix)
        {
            using (StreamWriter sw = File.CreateText(path))
            {
                var numVertexes = graphMatrix.GetLength(0);
                sw.WriteLine("{0}", numVertexes);

                for (int startVertex = 0; startVertex < numVertexes; startVertex++)
                {
                    for (int endVertex = 0; endVertex < numVertexes; endVertex++)
                    {
                        if (graphMatrix[startVertex, endVertex] != Config.emptyEdge)
                        {
                            sw.WriteLine("{0} {1} {2}"
                                , startVertex
                                , endVertex
                                , graphMatrix[startVertex, endVertex]);
                        }
                    }
                }
            }
        }

        public static int[,] ReadFromFile(string path)
        {
            using (StreamReader sr = new StreamReader(path))
            {
                int.TryParse(sr.ReadLine(), out var numVert);
                var graphMatrix = new int[numVert,numVert];

                for (int i = 0; i < numVert; i++)
                {
                    for (int j = 0; j < numVert; j++)
                    {
                        graphMatrix[i, j] = Config.emptyEdge;
                    }
                }

                while (sr.Peek() >= 0)
                {
                    //Parsing input triples
                    string[] edgeTriple = sr.ReadLine().Split(null);
                    int.TryParse(edgeTriple[0], out var startVertex);
                    int.TryParse(edgeTriple[1], out var endVertex);
                    int.TryParse(edgeTriple[2], out var weight);

                    //Initializing matrix
                    graphMatrix[startVertex, endVertex] = weight;
                }

                return graphMatrix;
            }
        }
    }
}
