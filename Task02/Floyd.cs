using System;
using System.Threading.Tasks;

namespace Task02
{
    class Floyd
    {
        private static void InitDistances(int[,] matrix, int[,] distances, int size)
        {
            Array.Copy(matrix, distances, matrix.Length);
            Parallel.For(0, size, i =>
            {
                for (int j = i; j < size; j++)
                    if (distances[i, j] == 0 && i != j)
                        distances[i, j] = distances[j, i] = Int32.MaxValue;
            });
        }

        public static int[,] Run(Graph graph)
        {
            var matrix = graph.GetMatrix();
            int size   = graph.VerticesNum;

            int[,] distances = new int[size, size];
            InitDistances(matrix, distances, size);

            Parallel.For(0, size, k =>
            {
                for (int i = 0; i < size; i++)
                    for (int j = 0; j < size; j++)
                    {
                        if (distances[i, j] > distances[i, k] + distances[k, j])
                            distances[i, j] = distances[i, k] + distances[k, j];
                    }
            });

            return distances;
        }
    }
}
