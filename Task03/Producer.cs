using System;
using System.Threading;

namespace Task03
{
    public class Producer
    {
        public Producer(CancellationToken ct)
        {
            Console.WriteLine("  Producer created in thread {0}", Thread.CurrentThread.ManagedThreadId);
            while(true) 
            {
                if (ct.IsCancellationRequested)
                {
                    Console.WriteLine("Task {0} cancelled", Thread.CurrentThread.ManagedThreadId);
                    ct.ThrowIfCancellationRequested();
                }

                Program.mutexIn.WaitOne();
                int delay = Produce();
                Program.semNotEmpty.Release();
                Program.mutexIn.ReleaseMutex();
                
                Thread.Sleep(1000 * delay);
            }
        }

        private int Produce()
        {
            int id = Thread.CurrentThread.ManagedThreadId;
            Data<int>.list.Add(id);
            Console.WriteLine("  Produced {0} by thread {1}", id, id);
            return id;
        }
    }
}