using System;

namespace Task02
{
    public class SequentialAlgoFloyd
    {
        public static int[,] Execute()
        {
            int n = GeneralResources.n;
            int[,] dist = new int[n, n];
            Array.Copy(GeneralResources.graphMatrix, dist, n * n);

            for (int k = 0; k < n; k++)
            {
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        if (dist[i, j] > dist[i, k] + dist[k, j])
                            dist[i, j] = dist[i, k] + dist[k, j];
                    }
                }
            }
            return dist;
        }
    }
}
