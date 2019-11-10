using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System;

namespace Task03 {
    public interface IWorker {
        Boolean isActive {
            get;
        }
    }
    public interface IConsumer<T>: IWorker {
        void Consume(T data);
    }

    public interface IProducer<T>: IWorker {
        IEnumerable<T> Produce();
    }

    public class Storage<T> {
        private readonly ConcurrentQueue<T> data = new ConcurrentQueue<T>();
        private readonly Mutex workMutex = new Mutex();
        private readonly SemaphoreSlim productionSemaphore = new SemaphoreSlim(0, Int32.MaxValue);
        private Boolean stopped = false;
        private Int32 consumersCount = 0;
        private Int32 producersCount = 0;

        public Storage() {
            workMutex.WaitOne();
        }

        public void Start() {
            if (!stopped) {
                workMutex.ReleaseMutex();
            }
        }

        public void Stop() {
            workMutex.WaitOne();

            stopped = true;

            while (producersCount != 0);

            while (productionSemaphore.CurrentCount != 0);

            Int32 waitCount = consumersCount;
            for (int i = 0; i < waitCount; i++) {
                productionSemaphore.Release();
            }

            while (consumersCount != 0);
        }

        private Thread StartWorker(Action action) {
            var thread = new Thread(() => {
                workMutex.WaitOne();
                workMutex.ReleaseMutex();
                action();
            });
            thread.Start();
            return thread;
        }

        public void AddConsumer(IConsumer<T> consumer) {
            var thread = StartWorker(() => {
                while (consumer.isActive) {
                    productionSemaphore.Wait();

                    T value;

                    if (data.TryDequeue(out value)) {
                        consumer.Consume(value);
                    } else if (stopped) {
                        break;
                    }
                }

                Interlocked.Decrement(ref consumersCount);
            });

            Interlocked.Increment(ref consumersCount);
        }

        public void AddProducer(IProducer<T> producer) {
            var thread = StartWorker(() => {
                while (producer.isActive && !stopped) {
                    var production = producer.Produce();

                    foreach (var value in production) {
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
