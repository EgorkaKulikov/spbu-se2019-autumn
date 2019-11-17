using System;
using System.Threading;

namespace Task03
{
    public class Producer<T> where T: new()
    {
    private volatile bool _isCancel = false;

    public Producer(string name)
    {
        var thread = new Thread(StartWork);
        thread.Name = "Produce thread " + name;
        Console.WriteLine(thread.Name);
        thread.Start();
    }

    private void StartWork()
    {
        while (!_isCancel)
        {
            Shared<T>.mutexProd.WaitOne();
            Shared<T>.List.Add(new T());
            Console.WriteLine($"Current list size after adding {Shared<T>.List.Count}");
            Shared<T>.full.Release();
            Shared<T>.mutexProd.ReleaseMutex();
            Thread.Sleep(2000);
        }
        
    }
    public void SetCancel()
    {
        _isCancel = true;
    }
    
    }
}