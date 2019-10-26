using System;

namespace Task03
{
    public class Consumer<T>
    {
        //volatile indicates that a field might be modified by multiple threads 
        private volatile bool isRunning = true;

        public void ReadData()
        {
            Console.WriteLine("Consumer starts working");
            while (isRunning)
            {
                Data<T>.fullSemaphore.WaitOne();
                Data<T>.mutex.WaitOne();
                //Reading data from queue
                Data<T>.buffer.Dequeue();
                Data<T>.emptySemaphore.Release();
                Data<T>.mutex.ReleaseMutex();
            }

            Console.WriteLine("Consuming loop exited");
        }

        public void StopRunning()
        {
            Console.WriteLine("Consumer stops working");
            isRunning = false;
        }
    }
}