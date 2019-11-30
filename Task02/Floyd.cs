using System;
using System.Threading;


public partial class Program
{  
  private static void Floyd(int[,] matrix, int size)
  {
    for (int k = 0; k < size; k++)
    {    
      var handles = new ManualResetEvent[size];
      
      for (int j = 0; j < size; j++)
      {
        handles[j] = new ManualResetEvent(false);

        ThreadPool.QueueUserWorkItem(new WaitCallback(stepFloyd), new object[] {k, j, size, matrix, handles[j]});
      }

      WaitHandle.WaitAll(handles);
    }
  }

  private static void stepFloyd(object state)
  {
    var array = state as object[];
    
    var k      = Convert.ToInt32(array[0]);
    var j      = Convert.ToInt32(array[1]);
    var size   = Convert.ToInt32(array[2]);
    var matrix = array[3] as int[,];
    var handle = array[4] as ManualResetEvent;

    for (int i = 0; i < size; i++)
    {
      if (0 == matrix[i, j] && i != j)
      {
        matrix[i, j] = matrix[i, k] + matrix[k, j];
      }
      else if (0 != matrix[i, k] + matrix[k, j])
      {
        matrix[i, j] = Math.Min(matrix[i, k] + matrix[k, j], matrix[i, j]);
      }
    }

    handle.Set();
  }
}
