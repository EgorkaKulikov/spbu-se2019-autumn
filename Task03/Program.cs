using System;
using System.Collections.Generic;
using System.Threading;

namespace Task03
{
    class Data<T>
    {
        public static List<T> buffer = new List<T>();
        public static Semaphore full = new Semaphore(0, Int32.MaxValue);
        public static Mutex prodMtx = new Mutex();
        public static Mutex consMtx = new Mutex();
    }
    class Producer<T> where T : new()
    {
        private bool _isCancelled = false;

        public void Cancel()
        {
            _isCancelled = true;
            Console.WriteLine($"Producer {Thread.CurrentThread.ManagedThreadId} is cancelled");
        }
        public Producer()
        {
            Thread pThread = new Thread(Produce);
            pThread.Start();
        }
        public void Produce()
        {
            while (!_isCancelled)
            {
                Data<T>.prodMtx.WaitOne();
                Data<T>.buffer.Add(new T());
                Console.WriteLine($"{Thread.CurrentThread.ManagedThreadId} is producing, #buffer = {Data<T>.buffer.Count}");
                Thread.Sleep(1000);
                Data<T>.prodMtx.ReleaseMutex();
                Data<T>.full.Release();
            }
        }
    }

    class Consumer<T> where T : new()
    {
        private bool _isCancelled = false;

        public void Cancel()
        { 
            _isCancelled = true;
            Console.WriteLine($"Consumer {Thread.CurrentThread.ManagedThreadId} is cancelled");
        }
        public Consumer()
        {
            Thread cThread = new Thread(Consume);
            cThread.Start();
        }

        public void Consume()
        {
            while (!_isCancelled)
            {
                Data<T>.full.WaitOne();
                if (_isCancelled)
                {
                    break;
                }
                Data<T>.consMtx.WaitOne();
                Data<T>.buffer.RemoveAt(0);
                Data<T>.consMtx.ReleaseMutex();
                Console.WriteLine($"{Thread.CurrentThread.ManagedThreadId} is consuming, #buffer = {Data<T>.buffer.Count}");
                Thread.Sleep(1000);
            }
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            int consCnt = 11;
            int prodCnt = 10;
            var producers = new List<Producer<int>>();
            var consumers = new List<Consumer<int>>();

            for (int i = 0; i < prodCnt; ++i)
            {
                producers.Add(new Producer<int>());
            }

            for (int i = 0; i < consCnt; ++i)
            {
                consumers.Add(new Consumer<int>());
            }

            Console.ReadKey();

            for (int i = 0; i < prodCnt; ++i)
            {
                producers[i].Cancel();
            }

            for (int i = 0; i < consCnt; ++i)
            {
                consumers[i].Cancel();
            }
        }
    }
}
