using System;
using System.Threading;

namespace Task03
{
    public class Consumer<T>
    {
        //volatile indicates that a field might be modified by multiple threads 
        private volatile bool isRunning = true;

        public void ReadData()
        {
            Console.WriteLine("Consumer starts , current thread id: {0}"
                , Thread.CurrentThread.ManagedThreadId
            );
            while (isRunning)
            {
                Data<T>.fullSemaphore.WaitOne();
                Data<T>.mutex.WaitOne();

                //Reading data from queue
                Data<T>.buffer.Dequeue();
                Console.WriteLine("Read from queue, num of elements: {0} current thread id: {1}"
                    , Data<T>.buffer.Count
                    , Thread.CurrentThread.ManagedThreadId
                );
                Thread.Sleep(Constants.TimeoutMs);

                Data<T>.emptySemaphore.Release();
                Data<T>.mutex.ReleaseMutex();
            }

            Console.WriteLine("Consuming loop exited, current thread id: {0}"
                , Thread.CurrentThread.ManagedThreadId
            );
        }

        public void StopRunning()
        {
            Console.WriteLine("Consumer stops working");
            isRunning = false;
        }
    }
}