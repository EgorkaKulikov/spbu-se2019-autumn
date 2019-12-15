namespace Task02
{
    class Kruskal
    {
// A class to represent a Subset for union-Find  
        public class Subset
        {
            public int Parent, Rank;
        };

// A utility function to Find set of an element i  
// (uses path compression technique)  
        static int Find(Subset[] subsets, int i)
        {
            // Find root and make root as  
            // parent of i (path compression)  
            if (subsets[i].Parent != i)
                subsets[i].Parent = Find(subsets,
                    subsets[i].Parent);

            return subsets[i].Parent;
        }

// A function that does union of  
// two sets of x and y (uses union by rank)  
        static void Union(Subset[] subsets, int x, int y)
        {
            int xroot = Find(subsets, x);
            int yroot = Find(subsets, y);

            // Attach smaller rank tree under root of 
            // high rank tree (Union by Rank)  
            if (subsets[xroot].Rank < subsets[yroot].Rank)
                subsets[xroot].Parent = yroot;
            else if (subsets[xroot].Rank > subsets[yroot].Rank)
                subsets[yroot].Parent = xroot;

            // If ranks are same, then make one as root  
            // and increment its rank by one  
            else
            {
                subsets[yroot].Parent = xroot;
                subsets[xroot].Rank++;
            }
        }

// The main function to construct MST  
// using Kruskal's algorithm  
        public static Graph.Edge[] RunSeqKruskal(int verticesNumber, Graph.Edge[] edgesList)
        {
            Graph.Edge[] result = new Graph.Edge[verticesNumber - 1]; // This will store the resultant MST  
            // Step 1: Sort all the edges in non-decreasing  
            // order of their weight. If we are not allowed  
            // to change the given graph, we can create 
            // a copy of array of edges  
            Sorts.StartQuickSortSeq(ref edgesList);
            // Allocate memory for creating V sSubsets  
            Subset[] subsets = new Subset[verticesNumber];
            // Create V Subsets with single elements  
            for (int v = 0; v < verticesNumber; ++v)
            {
                subsets[v] = new Subset();
                subsets[v].Parent = v;
                subsets[v].Rank = 0;
            }

            int e = 0; // An index variable, used for result[]  
            int i = 0; // Index used to pick next edge  

            // Number of edges to be taken is equal to V-1  
            while (e < verticesNumber - 1)
            {
                // Step 2: Pick the smallest edge. And increment  
                // the index for next iteration  
                Graph.Edge nextEdge = edgesList[i++];
                //  Console.WriteLine("{0} {1} {2}", nextEdge.src, nextEdge.dest, nextEdge.weight);

                int x = Find(subsets, nextEdge.src);
                int y = Find(subsets, nextEdge.dest);

                // If including this edge does't cause cycle,  
                // include it in result and increment the index  
                // of result for next edge  
                if (x != y)
                {
                    result[e++] = nextEdge;
                    Union(subsets, x, y);
                }

                // Else discard the nextEdge  
            }

            return result;
        }

        public static Graph.Edge[] RunParallelKruskal(int verticesNumber, Graph.Edge[] edgesList)
        {
            Graph.Edge[] result = new Graph.Edge[verticesNumber - 1]; // This will store the resultant MST  


            // Step 1: Sort all the edges in non-decreasing  
            // order of their weight. If we are not allowed  
            // to change the given graph, we can create 
            // a copy of array of edges  
            Sorts.StartQuickSortParallel(verticesNumber, ref edgesList);
            // Allocate memory for creating V Subsets  
            Subset[] subsets = new Subset[verticesNumber];
            // Create V Subsets with single elements  
            for (int v = 0; v < verticesNumber; ++v)
            {
                subsets[v] = new Subset();
                subsets[v].Parent = v;
                subsets[v].Rank = 0;
            }

            int e = 0; // An index variable, used for result[] 
            int i = 0; // Index used to pick next edge  

            // Number of edges to be taken is equal to V-1  
            while (e < verticesNumber - 1)
            {
                // Step 2: Pick the smallest edge. And increment  
                // the index for next iteration  
                Graph.Edge nextEdge = edgesList[i++];
                //  Console.WriteLine("{0} {1} {2}", nextEdge.src, nextEdge.dest, nextEdge.weight);

                int x = Find(subsets, nextEdge.src);
                int y = Find(subsets, nextEdge.dest);

                // If including this edge does't cause cycle,  
                // include it in result and increment the index  
                // of result for next edge  
                if (x != y)
                {
                    result[e++] = nextEdge;
                    Union(subsets, x, y);
                }

                // Else discard the nextEdge  
            }

            return result;
        }
    }
}