using System;
using System.Threading;


namespace Task02
{
  public partial class Program
  {
    private static int evalKruskal(Edge[] edgeList, int size)
    {
      var rand = new Random();
      sort(new object[] {edgeList, 0, edgeList.Length - 1, rand});

      var ostov = new int[size];

      int totalWeight      = 0;
      int indexOfComponent = 1;

      for (int i = 0; i < size; i++)
      {
        ostov[i] = 0;
      }

      for (int i = 0, f = 0; f < size - 1 && i < edgeList.Length; i++)
      {
        if (0 == ostov[edgeList[i].from] && 0 == ostov[edgeList[i].to])
        {
          ostov[edgeList[i].from] = indexOfComponent;
          ostov[edgeList[i].to]   = indexOfComponent;
          totalWeight += edgeList[i].weight;
          indexOfComponent++;
          f++;
        }
        else if (ostov[edgeList[i].from] == ostov[edgeList[i].to])
        {
          continue;
        }
        else if (0 == ostov[edgeList[i].from] && 0 != ostov[edgeList[i].to])
        {
          ostov[edgeList[i].from] = ostov[edgeList[i].to];
          totalWeight += edgeList[i].weight;
          f++;
        }
        else if (0 != ostov[edgeList[i].from] && 0 == ostov[edgeList[i].to])
        {
          ostov[edgeList[i].to] = ostov[edgeList[i].from];
          totalWeight += edgeList[i].weight;
          f++;
        }
        else if (0 != ostov[edgeList[i].from] && 0 != ostov[edgeList[i].to])
        {
          int min = Math.Min(ostov[edgeList[i].from], ostov[edgeList[i].to]);
          int max = Math.Max(ostov[edgeList[i].from], ostov[edgeList[i].to]);
          
          for (int j = 0; j < size; j++)
          {
            if (max == ostov[j])
            {
              ostov[j] = min;
            } 
          }

          totalWeight += edgeList[i].weight;
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
      var container = state as object[];

      var array = container[0] as Edge[];
      var left  = Convert.ToInt32(container[1]);
      var right = Convert.ToInt32(container[2]);
      var rand  = container[3] as Random;

      if (left < right)
      {
        int mid = partition(ref array, left, right, rand);
        
        if (256 < right - left)
        {
          var leftThr  = new Thread(new ParameterizedThreadStart(sort));
          var rightThr = new Thread(new ParameterizedThreadStart(sort));
          
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
}
