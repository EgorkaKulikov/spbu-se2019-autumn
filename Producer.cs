using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Configuration;
using System.Threading;

namespace Task03
{
    public class Producer<T> where T: new()
    {
    public volatile bool isCancel = false;

    public Producer(string name)
    {
        var thread = new Thread(StartWork);
        thread.Name = "Produce thread " + name;
        Console.WriteLine(thread.Name);
        thread.Start();
    }

    public void StartWork()
    {
        while (!isCancel)
        {
            Shared<T>.mutexProd.WaitOne();
            Shared<T>.list.Add(new T());
            Console.WriteLine($"Current list size after adding {Shared<T>.list.Count}");
            Shared<T>.full.Release();
            Shared<T>.mutexProd.ReleaseMutex();
            Thread.Sleep(2000);
        }
        
    }
    
    public void SetCancel()
    {
        isCancel = true;
    }
    
    }
}