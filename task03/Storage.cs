using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace Task03
{
    public interface IWorker
    {
        Boolean IsActive
        {
            get;
        }
    }
    public interface IConsumer<T> : IWorker
    {
        void Consume(T data);
    }

    public interface IProducer<T> : IWorker
    {
        IEnumerable<T> Produce();
    }

    public class Storage<T>
    {
        private readonly ConcurrentQueue<T> data = new ConcurrentQueue<T>();
        private readonly Mutex workMutex = new Mutex();
        private readonly SemaphoreSlim productionSemaphore = new SemaphoreSlim(0, Int32.MaxValue);
        private Boolean stopped = false;
        private Int32 consumersCount = 0;
        private Int32 producersCount = 0;

        public Storage()
        {
            workMutex.WaitOne();
        }

        public void Start()
        {
            workMutex.ReleaseMutex();
        }

        public void Stop()
        {
            workMutex.WaitOne();

            stopped = true;

            while (producersCount != 0) ;

            while (productionSemaphore.CurrentCount != 0) ;

            foreach (var _ in Enumerable.Range(1, consumersCount))
            {
                productionSemaphore.Release();
            }

            while (consumersCount != 0) ;

            stopped = false;
        }

        private Thread StartWorker(Action action)
        {
            var thread = new Thread(() =>
            {
                workMutex.WaitOne();
                workMutex.ReleaseMutex();
                action();
            });
            thread.Start();
            return thread;
        }

        public void AddConsumer(IConsumer<T> consumer)
        {
            var thread = StartWorker(() =>
            {
                while (consumer.IsActive)
                {
                    productionSemaphore.Wait();

                    if (data.TryDequeue(out T value))
                    {
                        consumer.Consume(value);
                    }
                    else if (stopped)
                    {
                        break;
                    }
                }

                Interlocked.Decrement(ref consumersCount);
            });

            Interlocked.Increment(ref consumersCount);
        }

        public void AddProducer(IProducer<T> producer)
        {
            var thread = StartWorker(() =>
            {
                while (producer.IsActive && !stopped)
                {
                    var production = producer.Produce();

                    foreach (var value in production)
                    {
                        data.Enqueue(value);
                        productionSemaphore.Release();
                    }
                }

                Interlocked.Decrement(ref producersCount);
            });

            Interlocked.Increment(ref producersCount);
        }
    }
}
