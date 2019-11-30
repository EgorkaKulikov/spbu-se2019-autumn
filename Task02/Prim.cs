using System;
using System.Linq;
using System.Threading.Tasks;
using System.Collections.Generic;


public partial class Program
{
  private static int Prim(int[,] matrix, int size)
  {
    Edge[] edges = new Edge[size];
    int[] nodes = new int[size];
    nodes[0] = 0;

    int totalWeight = 0;

    for (int i = 1; i < size; i++)
    {

      Task<Edge>[] searches = new Task<Edge>[i];

      Parallel.For(0, i, j =>
      {
        searches[j] = Task<Edge>.Run(() => findMin(new object[] {matrix, size, nodes[j], nodes}));    
      });

      Task<Edge>.WaitAll(searches);
      
      Edge min = new Edge(0, 0, 0);

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
    object[] container = state as object[];
    int[,] matrix = container[0] as int[,];
    int size = Convert.ToInt32(container[1]);
    int node = Convert.ToInt32(container[2]);
    int[] nodes = container[3] as int[];

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

    Edge buf = new Edge(node, index, weight);
    
    return Task<Edge>.FromResult(buf);
  }
}
