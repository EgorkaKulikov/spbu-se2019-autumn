using System;
using System.Collections.Generic;
using System.Security.Cryptography.X509Certificates;
using System.Threading;

namespace Task03
{
    public class Consumer<T>
    { 
        private bool isCansel = false;
        
    public Consumer(string name)
    {
        var thread = new Thread(StartWork);
        thread.Name = "Consumer thread " + name;
        Console.WriteLine(thread.Name);
        thread.Start();
    }

    public void StartWork()
    {
        while (!isCansel)
        {
            Shared<T>.mutexCons.WaitOne();
            Shared<T>.full.WaitOne();
            Shared<T>.list.RemoveAt(Shared<T>.list.Count - 1);
            Console.WriteLine($"Current list size after removing {Shared<T>.list.Count}");
            Shared<T>.mutexCons.ReleaseMutex();
            Thread.Sleep(2000);
        }
    }

    public void setCansel()
    {
        isCansel = true;
    }

    }
}