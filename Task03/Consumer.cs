using System;
using System.Threading;

namespace Task03
{
    public class Consumer<T>
    { 
        private volatile bool _isCancel = false;
        
    public Consumer(string name)
    {
        var thread = new Thread(StartWork);
        thread.Name = "Consumer thread " + name;
        Console.WriteLine(thread.Name);
        thread.Start();
    }

    private void StartWork()
    {
        while (!_isCancel)
        {
            Shared<T>.mutexCons.WaitOne();
            
            if (_isCancel)
            {
                Shared<T>.mutexCons.ReleaseMutex();
                break;
            }

            Shared<T>.full.WaitOne();
            Shared<T>.List.RemoveAt(Shared<T>.List.Count - 1);
            Console.WriteLine($"Current list size after removing {Shared<T>.List.Count}");
            Shared<T>.mutexCons.ReleaseMutex();
            
            Thread.Sleep(2000);
        }
    }

    public void SetCancel()
    {
        _isCancel = true;
    }

    }
}