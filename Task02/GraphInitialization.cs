using System;
using System.Collections.Generic;
using System.IO;

namespace Graphs
{
    public class Graph
    {
        public Graph()
        {
            _random = new Random();
            Size = _random.Next(Constants.MinNodes, Constants.MaxNodes + 1);
            AmountEdges = _random.Next(Constants.MinEdges, Size * (Size - 1) / 2 + 1);
        }
        
        public int AmountEdges { get; private set; }
        public int Size { get; private set; }

        private readonly Random _random;

        public int[,] GenerateNonEdgesGraph() => GenerateGraph(0);

        public int[,] GenerateGraphWithEdges() => GenerateGraph(AmountEdges);
        
        private int[,] GenerateGraph(int edgesLimit)
        {
            int[,] graph = new int[Size, Size];

            for (int row = 0; row < Size; ++row)
            {
                for (int column = row; column < Size; ++column)
                {
                    graph[row, column] = Constants.Inf;
                    graph[column, row] = graph[row, column];
                }
            }
            
            while (edgesLimit-- != 0)
            {
                int row = _random.Next(Size);
                int column = _random.Next(Size);
                
                while (graph[row, column] != Constants.Inf
                       || row == column)
                {
                    row = _random.Next(Size);
                    column = _random.Next(Size);
                }

                graph[row, column] = _random.Next(Constants.MinEdgeValue
                                                , Constants.MaxEdgeValue + 1);
                graph[column, row] = graph[row, column];
            }
            return graph;
        }

        public int[,] FromListEdgesToMatrix(List<Edge> edges)
        {
            int[,] newGraph = GenerateNonEdgesGraph();

            foreach (var edge in edges)
            {
                newGraph[edge.begin, edge.end] = edge.weight;
                newGraph[edge.end, edge.begin] = edge.weight;
            }
            
            return newGraph;
        }

        public List<Edge> FromMatrixToListEdges(int[,] graph, int size)
        {
            List<Edge> edges = new List<Edge>();
            
            for (int row = 0; row < size; ++row)
            {
                for (int column = row + 1; column < size; ++column)
                {
                    if (graph[row, column] != Constants.Inf)
                        edges.Add(new Edge(row, column, graph[row, column]));
                }
            }

            return edges;
        }
        
        public void PrintGraph(int[,] graph, int size, StreamWriter writer)
        {
            for (int row = 0; row < size; ++row)
            {
                for (int column = 0; column < size; ++column)
                {
                    writer.Write(graph[row, column] == Constants.Inf ? "0   " : $"{graph[row, column]}   ");
                }
                writer.WriteLine();
            }
            
            writer.WriteLine();
        }

        public bool CompareMatrixGraphs(int[,] firstGraph, int[,] secondGraph, int size)
        {
            for (int row = 0; row < size; ++row)
            {
                for (int column = 0; column < size; ++column)
                {
                    if (firstGraph[row, column] != secondGraph[row, column])
                        return false;
                }
            }

            return true;
        }
    }

    public class Edge : IComparable<Edge>
    {
        public int begin;
        public int end;
        public int weight;

        public Edge(int begin, int end, int weight)
        {
            this.begin = begin;
            this.end = end;
            this.weight = weight;
        }

        public int CompareTo(Edge other) => this.weight.CompareTo(other.weight);
    }
}