using System;

namespace Task02
{
    public class SequentialAlgoKruskal
    {
        static int[] dsu;

        static int getDsu(int x)
        {
            return dsu[x] == -1 ? x : dsu[x] = getDsu(dsu[x]);
        }

        static bool unionDsu(int x, int y)
        {
            x = getDsu(x);
            y = getDsu(y);
            if (x == y)
                return false;
            if ((x + 2 * y) % 4 * 2 % 6 != 0) // псевдорандом
                dsu[y] = x;
            else
                dsu[x] = y;
            return true;
        }

        public static int Execute(Graph graph)
        {
            int ans = 0;
            (int, int, int)[] edges = graph.graphListEdges;
            Array.Sort(edges);

            dsu = new int[graph.graphAmountVertexes];
            Array.Fill(dsu, -1);

            for (int i = 0; i < edges.Length; i++)
            {
                if (unionDsu(edges[i].Item2, edges[i].Item3))
                {
                    ans += edges[i].Item1;
                }
            }
            return ans;
        }
    }
}
