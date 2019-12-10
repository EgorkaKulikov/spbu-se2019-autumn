using System;
using System.Threading;

namespace Task03
{
    public class Consumer<T>
    {
        private static bool IsGetting = true;
        public static void StopGetting() => IsGetting = false;
        
        public void GetData()
        {
            while (IsGetting)
            {
                SharedRes<T>.Empty.WaitOne();

                if (!IsGetting)
                {
                    break;
                }
                
                SharedRes<T>.MConsumer.WaitOne();
                
                T removed = SharedRes<T>.Data[^1];
                SharedRes<T>.Data.RemoveAt(SharedRes<T>.Data.Count - 1);
                Console.WriteLine($"Consumer withdrew {removed} by thread {Thread.CurrentThread.ManagedThreadId}");
                
                Thread.Sleep(SharedRes<T>.TimeoutConsumer);

                SharedRes<T>.MConsumer.ReleaseMutex();
            }
        }
    }
}