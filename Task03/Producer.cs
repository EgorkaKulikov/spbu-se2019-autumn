using System;
using System.Threading;

namespace ProducerConsumer
{
    public class Producer<T> where T : new()
    {
        private bool isRunning = true;

        public void StopRunning()
        {
            isRunning = false;
        }

        public void ProduceData()
        {
            while (isRunning)
            {
                Data<T>.Mutex.WaitOne();
                if (!isRunning)
                {
                    Data<T>.Mutex.ReleaseMutex();
                    break;
                }
                Data<T>.Buffer.Add(new T());
                Console.WriteLine($"Producer {Thread.CurrentThread.ManagedThreadId} has wrote to buffer");
                Thread.Sleep(Data<T>.Timeout);
                Data<T>.Empty.Release();
                Data<T>.Mutex.ReleaseMutex();
            }
            Console.WriteLine($"Producer {Thread.CurrentThread.ManagedThreadId} has stoped");
        }
    }
}