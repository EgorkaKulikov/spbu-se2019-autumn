using System;
using System.Collections.Generic;
using System.Threading;

namespace Task_03
{
    class Data<T>
    {
        public static readonly List<T> DataList = new List<T>();
        public static readonly Semaphore empty = new Semaphore(0, Int32.MaxValue);
        public static readonly Semaphore full = new Semaphore(Int32.MaxValue, Int32.MaxValue);
        public static readonly Mutex ConsumerMutex = new Mutex();
        public static readonly Mutex ProducerMutex = new Mutex();
    }

    class Producer<T> where T : new()
    {
        private volatile bool _working = true;
        public void StopWorking() => _working = false;

        public void Produce ()
        {
            Console.WriteLine("Producer {0} starts working", Thread.CurrentThread.ManagedThreadId);
            while (_working)
            {
                Data<T>.full.WaitOne();
                Data<T>.ProducerMutex.WaitOne();

                Data<T>.DataList.Add(new T());
                Console.WriteLine("Producer {0} adds new data", Thread.CurrentThread.ManagedThreadId);

                Data<T>.ProducerMutex.ReleaseMutex();
                Data<T>.empty.Release();
                Thread.Sleep(1000);
            }
            Console.WriteLine("Producer {0} stops working", Thread.CurrentThread.ManagedThreadId);
        }
    }

    class Consumer<T>
    {
        private volatile bool _working = true;
        public void StopWorking() => _working = false;

        public void Consume()
        {
            Console.WriteLine("Consumer {0} starts working", Thread.CurrentThread.ManagedThreadId);
            while (_working)
            {
                Data<T>.empty.WaitOne();
                if (!_working)
                {
                    break;
                }
                Data<T>.ConsumerMutex.WaitOne();

                var data = Data<T>.DataList[0];
                Data<T>.DataList.RemoveAt(0);
                Console.WriteLine("Consumer {0} reads {1}", Thread.CurrentThread.ManagedThreadId, data);

                Data<T>.full.Release();
                Data<T>.ConsumerMutex.ReleaseMutex();
                Thread.Sleep(1000);
            }
            Console.WriteLine("Producer {0} stops working", Thread.CurrentThread.ManagedThreadId);
        }
    }

    internal static class Program
    {
        private const int ConsumerNumber = 5;
        private const int ProducerNumber = 5;

        public static void Main(string[] args)
        {
            var consumers = new List<Consumer<int>>();
            var producers = new List<Producer<int>>();

            for (var num = 0; num < ProducerNumber; num++)
            {
                var producer = new Producer<int>();
                producers.Add(producer);
                var producerThread = new Thread(producer.Produce);
                producerThread.Start();
            }

            for (var num = 0; num < ConsumerNumber; num++)
            {
                var consumer = new Consumer<int>();
                consumers.Add(consumer);
                var consumerThread = new Thread(consumer.Consume);
                consumerThread.Start();
            }

            Console.ReadKey();

            foreach (var consumer in consumers) consumer.StopWorking();
            foreach (var producer in producers) producer.StopWorking();

            for(int i=0; i < ConsumerNumber; i++)
            {
                Data<int>.empty.Release();
            }
        }
    }
}