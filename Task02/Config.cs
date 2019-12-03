using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Task02
{
    public static class Config
    {
        //Output config
        public const string graphPath = "test_graph.txt";
        public const string resultsKruskalPath = "results_kruskal.txt";
        public const string resultsPrimPath = "results_prim.txt";
        public const string resultsFloydPath = "results_floyd.txt";

        public const int numVertexes = 5000;
        public const int maxWeight = 100;
        public const int numEdges = 100000;
        public const int emptyEdge = -1;
        public const int chunkSize = 100;
    }
}
