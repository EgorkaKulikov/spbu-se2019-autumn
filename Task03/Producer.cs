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
                Data<T>.BufMutex.WaitOne();

                if (isRunning)
                {
                    //Adding data to queue
                    Data<T>.Buffer.Enqueue(new T());
                    Data<T>.WriteCnt++;
                    Console.WriteLine("Added to queue,  num of elements: {0} current thread id: {1}"
                        , Data<T>.Buffer.Count
                        , Thread.CurrentThread.ManagedThreadId
                    );

                    //Timeout each specified TimeOutIterations
                    if (0 == Data<T>.WriteCnt % Constants.TimeoutIterations)
                    {
                        Thread.Sleep(Constants.TimeoutMs);
                    }
                }

                Data<T>.BufMutex.ReleaseMutex();
                Data<T>.BufSemaphore.Release();
            }
        }

        public void StopRunning()
        {
            Console.WriteLine("Producer stops working");
            isRunning = false;
        }
    }
}