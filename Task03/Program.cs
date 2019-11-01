using System;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;

namespace Task03
{
  class Program
  {
    private static Mutex mList = new Mutex();
    private static Mutex mCons = new Mutex();
    private static Mutex mProd = new Mutex();
    private static List<int> list = new List<int>();
    private static int pauseProd, pauseCons;
    private static int numOfProd, numOfCons;

    public static void Main(string[] args)
    {
      Console.Write("Enter number of producers: ");
      
      while (false == Int32.TryParse(Console.ReadLine(), out numOfProd) || 0 >= numOfProd)
      {
        Console.Write("Incorrect input. Try natural number: ");
      }
      
      Console.Write("Enter number of consumers: ");
      
      while (false == Int32.TryParse(Console.ReadLine(), out numOfCons) || 0 >= numOfCons)
      {
        Console.Write("Incorrect input. Try natural number: ");
      }

      CancellationTokenSource source = new CancellationTokenSource();
      List<Task> tasks = new List<Task>();
      TaskFactory factory = new TaskFactory(source.Token);

      pauseProd = 0;
      pauseCons = 0;

      for (int i = 0; i < numOfProd; i++)
      {
        tasks.Add(factory.StartNew( () => produce(source.Token)));
      }
      
      for (int i = 0; i < numOfCons; i++)
      {
        tasks.Add(factory.StartNew( () => consume(source.Token)));
      }
      
      Console.ReadKey();
      Console.WriteLine("\nCancellation requested.");
      source.Cancel();
      Task.WaitAll(tasks.ToArray());
      source.Dispose();
    }

    private static int produce(CancellationToken ct)
    {
      while (true)
      {
        mProd.WaitOne();
        
        if (0 == pauseProd)
        {
          Thread.Sleep(100);
        }
        
        mList.WaitOne();
        
        var item = list.Count;
        list.Add(item);
        
        mList.ReleaseMutex();
        
        pauseProd = (pauseProd + 1) % 2;
        Console.WriteLine($"Thread {Thread.CurrentThread.ManagedThreadId} write {item}.");
        
        mProd.ReleaseMutex();
        
        if (true == ct.IsCancellationRequested)
        {
          Console.WriteLine($"Thread-producer {Thread.CurrentThread.ManagedThreadId} cancalled.");
          numOfProd--;
          
          return 0;
        }
      }
    }

    private static int consume(CancellationToken ct)
    {
      while (true)
      {
        mCons.WaitOne();
        
        if (0 == pauseCons)
        {
          Thread.Sleep(100);
        }
        
        if (0 != list.Count)
        {
          mList.WaitOne();
        
          var item = list[list.Count - 1];
          list.RemoveAt(list.Count - 1);
          pauseCons = (pauseCons + 1) % 2;
        
          mList.ReleaseMutex();
        
          Console.WriteLine($"Thread {Thread.CurrentThread.ManagedThreadId} read {item}.");
        }
        
        mCons.ReleaseMutex();
        
        if (true == ct.IsCancellationRequested && 0 == list.Count && 0 == numOfProd)
        {
          Console.WriteLine($"Thread-consumer {Thread.CurrentThread.ManagedThreadId} cancelled.");
          numOfCons--;

          return 0;
        }
      }
    }
  }
}
