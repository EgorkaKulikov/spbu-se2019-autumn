using System;
using System.Linq;
using System.Threading;

namespace Task03
{
    public class Consumer
    {
        public Consumer(CancellationToken ct)
        {
            Console.WriteLine("  Consumer created in thread {0}", Thread.CurrentThread.ManagedThreadId);
            while (true) 
            {
                Program.semNotEmpty.Wait();
                if (ct.IsCancellationRequested)
                {
                    Console.WriteLine("Task {0} cancelled", Thread.CurrentThread.ManagedThreadId);
                    ct.ThrowIfCancellationRequested();
                }
                Program.mutexIn.WaitOne();
                int delay = Consume();
                Program.mutexIn.ReleaseMutex();
                Thread.Sleep(1000 * delay);
            }
        }

        private int Consume()
        {
            int id;
            if (Data<int>.list.Any())
            {
                id = Data<int>.list[0];
                Data<int>.list.RemoveAt(0);
            }
            else
            {
                id = -1;
            }
            Console.WriteLine("  Consumed {0} by thread {1}", id, Thread.CurrentThread.ManagedThreadId);
            return id;
        }
    }
}