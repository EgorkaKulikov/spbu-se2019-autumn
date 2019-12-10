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
                Data<T>.BufSemaphore.WaitOne();
                Data<T>.BufMutex.WaitOne();

                //Reading data from queue
                if (isRunning)
                {
                    Data<T>.ReadCnt++;
                    Data<T>.Buffer.Dequeue();
                    Console.WriteLine("Read from queue, num of elements: {0} current thread id: {1}"
                        , Data<T>.Buffer.Count
                        , Thread.CurrentThread.ManagedThreadId
                    );

                    //Timeout each specified TimeOutIterations
                    if (0 == Data<T>.ReadCnt % Constants.TimeoutIterations)
                    {
                        Thread.Sleep(Constants.TimeoutMs);
                    }
                }

                Data<T>.BufMutex.ReleaseMutex();
            }
        }

        public void StopRunning()
        {
            Console.WriteLine("Consumer stops working");
            isRunning = false;
        }
    }
}