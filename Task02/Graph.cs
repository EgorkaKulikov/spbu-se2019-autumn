using System;
using System.Collections.Generic;

namespace Task02
{
    class Graph
    {
        #region Exception messages
        private string MatrixNotSquareExceptionMessage        = "Adjacency matrix must be square!";
        private string MatrixWasNullExceptionMessage          = "Adjacency matrix does not exist!";
        private string MatrixIsNotSymmetricalExceptionMessage = "Adjacency matrix is not symmetrical!";
        #endregion

        private List<Edge> Edges = new List<Edge>();

        private readonly int[,] Matrix;

        public int VerticesNum { get; private set; } = 0;

        public Graph(int[,] matrix)
        {
            CheckInput(matrix);

            Matrix = matrix;
            VerticesNum = matrix.GetLength(0);

            BuildEdges();
        }

        public List<Edge> GetEdges()
        {
            return new List<Edge>(Edges);
        }

        public int[,] GetMatrix()
        {
            int[,] matrix = new int[VerticesNum, VerticesNum];
            for (int i = 0; i < VerticesNum; i++)
                for (int j = i + 1; j < VerticesNum; j++)
                {
                    matrix[i, j] = matrix[j, i] = Matrix[i, j];
                }
            return matrix;
        }

        private void CheckInput(int[,] matrix)
        {
            if (matrix == null)
                throw new NullReferenceException(MatrixWasNullExceptionMessage);

            if (matrix.GetLength(0) != matrix.GetLength(1))
                throw new Exception(MatrixNotSquareExceptionMessage);

            var matrixSize = matrix.GetLength(0);
            bool wasError = false;

            for (int i = 0; i < matrixSize; i++)
                for (int j = 0; j < matrixSize; j++)
                    if (matrix[i, j] != matrix[j, i] ||
                        i == j && matrix[i, j] != 0)
                    {
                        wasError = true;
                        break;
                    }

            if (wasError)
                throw new Exception(MatrixIsNotSymmetricalExceptionMessage);
        }

        private void BuildEdges()
        {
            var matrixSize = Matrix.GetLength(0);

            for (int i = 0; i < matrixSize - 1; i++)
                for (int j = i + 1; j < matrixSize; j++)
                {
                    if (Matrix[i, j] != 0)
                        Edges.Add(new Edge(i, j, Matrix[i, j]));
                }
        }
    }
}
