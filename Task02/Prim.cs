using System;
using System.Linq;
using System.Threading.Tasks;
using System.Collections.Generic;


namespace Task02
{
  public partial class Program
  { 
    private static int evalPrim(int[,] matrix, int size)
    {
      var edges = new Edge[size];
      var nodes = new int[size];
      
      nodes[0] = 0;
      int totalWeight = 0;

      for (int i = 1; i < size; i++)
      {

        var searches = new Task<Edge>[i];

        Parallel.For(0, i, j =>
        {
          searches[j] = Task<Edge>.Run(() => findMin(new object[] {matrix, size, nodes[j], nodes}));    
        });

        Task<Edge>.WaitAll(searches);
        
        var min = new Edge(0, 0, 0);

        for (int j = 0; j < i; j++)
        {
          edges[j] = searches[j].Result;
        }

        for (int j = 0; j < i; j++)
        {
          if (0 == min.weight
          || (min.weight > edges[j].weight
          &&  0 != edges[j].weight))
          {
            swap(ref min, ref edges[j]);
          }
        }

        totalWeight += min.weight;

        matrix[min.from, min.to] = 0;
        matrix[min.to, min.from] = 0;

        if (nodes.Contains(min.from))
        {
          nodes[i] = min.to;
        }
        else
        {
          nodes[i] = min.from;
        }
      }

      return totalWeight;
    }

    private static Task<Edge> findMin(object state)
    {
      var container = state as object[];
      
      var matrix = container[0] as int[,];
      var size   = Convert.ToInt32(container[1]);
      var node   = Convert.ToInt32(container[2]);
      var nodes  = container[3] as int[];

      int index  = 0;
      int weight = 0;

      for (int i = 0; i < size; i++)
      {
        if (!nodes.Contains(i) 
        && (0 == weight 
        || (0 != matrix[node, i] 
        && matrix[node, i] < weight)))
        {
          weight = matrix[node, i];
          index = i;
        }
      }

      var buf = new Edge(node, index, weight);
      
      return Task<Edge>.FromResult(buf);
    }
  }
}
