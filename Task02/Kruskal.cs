using System.IO;

namespace Task02
{
    public static class Kruskal
    {
        private class Dsu
        {
            internal readonly int[] Parent;
            internal readonly int[] Rank;

            public Dsu(int verticesCount)
            {
                Parent = new int[verticesCount];
                Rank = new int[verticesCount];

                for (var i = 0; i < verticesCount; i++)
                {
                    Parent[i] = i;
                    Rank[i] = 0;
                }
            }

            internal int Find(int i)
            {
                if (i == Parent[i])
                {
                    return i;
                }
                Parent[i] = Find(Parent[i]);
                return Parent[i];
            }

            internal void Union(int i, int j)
            {
                i = Find(i);
                j = Find(j);

                if (i == j)
                {
                    return;
                }

                if (Rank[i] > Rank[j])
                {
                    Parent[j] = i;
                }
                else
                {
                    Parent[i] = j;
                    if (Rank[i] == Rank[j])
                    {
                        Rank[j]++;
                    }
                }
            }
        }

        private static int ParallelKruskal(this Graph graph)
        {
            var dsu = new Dsu(graph.VerticesCount);
            var edges = graph.EdgesList.ToArray();
            for (int v = 0; v < graph.VerticesCount; v++)
            {
                dsu.Parent[v] = v;
                dsu.Rank[v] = 0;
            }
            
            Sort.QSortParallel(edges, 0 , edges.Length);
            
            int i = 0, cnt = 0, weight = 0;
            
            while (cnt < graph.VerticesCount - 1)
            {
                Graph.Edge nextEdge = edges[i++];

                int x = dsu.Find(nextEdge.End);
                int y = dsu.Find(nextEdge.Start);

                if (x != y)
                {
                    edges[cnt++] = nextEdge;
                    dsu.Union( x, y);
                    weight += nextEdge.Weight;
                }
            }
            return weight;
        }
        
        public static void PrintKruskal(this Graph graph)
        {
            using var writer = new StreamWriter("Kruskal.txt");
            int result = graph.ParallelKruskal();
            writer.Write($"{result} ");
        }
    }
}
