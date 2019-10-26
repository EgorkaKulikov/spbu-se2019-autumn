using System;

namespace Task03
{
    public class Producer<T>
        where T : new()
    {
        //volatile indicates that a field might be modified by multiple threads 
        private volatile bool isRunning = true;

        public void WriteData()
        {
            Console.WriteLine("Producer starts working");
            while (isRunning)
            {
                Data<T>.emptySemaphore.WaitOne();
                Data<T>.mutex.WaitOne();
                //Adding data to queue
                Data<T>.buffer.Enqueue(new T());
                Data<T>.mutex.ReleaseMutex();
                Data<T>.fullSemaphore.Release();
            }
            
            Console.WriteLine("Producing loop exited");
        }

        public void StopRunning()
        {
            Console.WriteLine("Producer stops working");
            isRunning = false;
        }
    }
}