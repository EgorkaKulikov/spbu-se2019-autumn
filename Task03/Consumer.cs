using System;
using System.Threading;

namespace ProducerConsumer
{
    public class Consumer<T>
    {
        private bool isRunning = true;

        public void StopRunning()
        {
            isRunning = false;
        }

        public void ConsumeData()
        {
            while (isRunning)
            {
                Data<T>.Empty.WaitOne();
                if (!isRunning)
                {
                    break;
                }
                Data<T>.Mutex.WaitOne();
                Data<T>.Buffer.RemoveAt(0);
                Console.WriteLine($"Consumer {Thread.CurrentThread.ManagedThreadId} has wrote to buffer");
                Thread.Sleep(Data<T>.Timeout);
                Data<T>.Mutex.ReleaseMutex();
            }
            Console.WriteLine($"Consumer {Thread.CurrentThread.ManagedThreadId} has stoped..");
        }
    }
}