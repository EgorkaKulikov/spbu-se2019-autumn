using System;
using System.Threading;

namespace Task03
{
    public class Producer<T>
        where T : new()
    {
        //volatile indicates that a field might be modified by multiple threads 
        private volatile bool isRunning = true;

        public void WriteData()
        {
            Console.WriteLine("Producer starts working, current thread id: {0}"
                , Thread.CurrentThread.ManagedThreadId
            );
            while (isRunning)
            {
                Data<T>.emptySemaphore.WaitOne();
                Data<T>.mutex.WaitOne();

                //Adding data to queue
                Data<T>.buffer.Enqueue(new T());
                Console.WriteLine("Added to queue,  num of elements: {0} current thread id: {1}"
                    , Data<T>.buffer.Count
                    , Thread.CurrentThread.ManagedThreadId
                );
                Thread.Sleep(Constants.TimeoutMs);

                Data<T>.mutex.ReleaseMutex();
                Data<T>.fullSemaphore.Release();
            }

            Console.WriteLine("Producing loop exited, current thread id: {0}"
                , Thread.CurrentThread.ManagedThreadId
            );
        }

        public void StopRunning()
        {
            Console.WriteLine("Producer stops working");
            isRunning = false;
        }
    }
}