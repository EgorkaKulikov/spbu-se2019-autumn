using System;
using System.Threading;


public partial class Program
{
  private static int Kruskal(Edge[] smezhn, int size)
  {
    Random rand = new Random();
    sort(new object[] {smezhn, 0, smezhn.Length - 1, rand});

    int[] ostov = new int[size];

    int totalWeight = 0;
    int newComp = 1;

    for (int i = 0; i < size; i++)
    {
      ostov[i] = 0;
    }

    for (int i = 0, f = 0; f < size - 1 && i < smezhn.Length; i++)
    {
      if (0 == ostov[smezhn[i].from] && 0 == ostov[smezhn[i].to])
      {
        ostov[smezhn[i].from] = newComp;
        ostov[smezhn[i].to]   = newComp;
        totalWeight += smezhn[i].weight;
        newComp++;
        f++;
      }
      else if (ostov[smezhn[i].from] == ostov[smezhn[i].to])
      {
        continue;
      }
      else if (0 == ostov[smezhn[i].from] && 0 != ostov[smezhn[i].to])
      {
        ostov[smezhn[i].from] = ostov[smezhn[i].to];
        totalWeight += smezhn[i].weight;
        f++;
      }
      else if (0 != ostov[smezhn[i].from] && 0 == ostov[smezhn[i].to])
      {
        ostov[smezhn[i].to] = ostov[smezhn[i].from];
        totalWeight += smezhn[i].weight;
        f++;
      }
      else if (0 != ostov[smezhn[i].from] && 0 != ostov[smezhn[i].to])
      {
        int min = Math.Min(ostov[smezhn[i].from], ostov[smezhn[i].to]);
        int max = Math.Max(ostov[smezhn[i].from], ostov[smezhn[i].to]);
        
        for (int j = 0; j < size; j++)
        {
          if (max == ostov[j])
          {
            ostov[j] = min;
          } 
        }

        totalWeight += smezhn[i].weight;
        f++;
      }
    }

    return totalWeight;
  }

  private static int partition(ref Edge[] array, int left, int right, Random rand)
  {
    int pivotIndex = rand.Next(left, right);
    int pivotValue = array[pivotIndex].weight;
    int j = left;

    swap(ref array[pivotIndex], ref array[right]);

    for (int i = left; i < right; i++)
    {
      if (array[i].weight < pivotValue)
      {
        swap(ref array[i], ref array[j]);
        j++;
      }
    }

    swap(ref array[j], ref array[right]);

    return j;
  }

  private static void sort(object state)
  {
    object[] container = state as object[];
    Edge[] array = container[0] as Edge[];
    int left = Convert.ToInt32(container[1]);
    int right = Convert.ToInt32(container[2]);
    Random rand = container[3] as Random;

    if (left < right)
    {
      int mid = partition(ref array, left, right, rand);
      
      if (256 < right - left)
      {
        Thread leftThr  = new Thread(new ParameterizedThreadStart(sort));
        Thread rightThr = new Thread(new ParameterizedThreadStart(sort));
        
        leftThr.Start(new object[] {array, left, mid - 1, rand});
        rightThr.Start(new object[] {array, mid + 1, right, rand});

        leftThr.Join();
        rightThr.Join();
      }
      else if (1 < right - left)
      { 
        sort(new object[] {array, left, mid - 1, rand});
        sort(new object[] {array, mid + 1, right, rand});
      }
    }
  }  
}
