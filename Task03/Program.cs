using System;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;

namespace Task03
{
  class Program
  {
    private static Mutex mut = new Mutex();
    private static Mutex mut2 = new Mutex();
    private static Mutex mut3 = new Mutex();
    private static List<int> list = new List<int>();
    private static int pauseCons, pauseProd;

    public static void Main(string[] args)
    {
      int numOfProd;
      Console.Write("Enter number of producers: ");
      
      while (!Int32.TryParse(Console.ReadLine(), out numOfProd) || 0 >= numOfProd)
      {
        Console.Write("Incorrect input. Try natural number: ");
      }
      
      int numOfCons;
      Console.Write("Enter number of consumers: ");
      
      while (!Int32.TryParse(Console.ReadLine(), out numOfCons) || 0 >= numOfCons)
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
        mut3.WaitOne();
        
        if (0 == pauseProd)
        {
          Thread.Sleep(100);
        }
        
        mut.WaitOne();
        
        var item = list.Count;
        list.Add(item);
        
        mut.ReleaseMutex();
        
        pauseProd = (pauseProd + 1) % 2;
        Console.WriteLine($"Thread {Thread.CurrentThread.ManagedThreadId} write {item}.");
        
        mut3.ReleaseMutex();
        
        if (true == ct.IsCancellationRequested)
        {
          Console.WriteLine($"Thread {Thread.CurrentThread.ManagedThreadId} cancalled.");
          return 0;
        }
      }
    }

    private static int consume(CancellationToken ct)
    {
      while (true)
      {
        mut2.WaitOne();
        
        if (0 == pauseCons)
        {
          Thread.Sleep(100);
        }
        
        if (0 != list.Count)
        {
          mut.WaitOne();
        
          var item = list[list.Count - 1];
          list.RemoveAt(list.Count - 1);
          pauseCons = (pauseCons + 1) % 2;
        
          mut.ReleaseMutex();
        
          Console.WriteLine($"Thread {Thread.CurrentThread.ManagedThreadId} read {item}.");
        
          mut2.ReleaseMutex();
        } 
        else
        {
          Thread.Sleep(1000);
        
          mut2.ReleaseMutex();
        }
        
        if (true == ct.IsCancellationRequested)
        {
          Console.WriteLine($"Thread {Thread.CurrentThread.ManagedThreadId} cancelled.");
        
          return 0;
        }
      }
    }
  }
}
