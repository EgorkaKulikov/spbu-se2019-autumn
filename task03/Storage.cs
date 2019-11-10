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
        private readonly ConcurrentDictionary<IConsumer<T>, Thread> consumers = new ConcurrentDictionary<IConsumer<T>, Thread>();
        private readonly ConcurrentDictionary<IProducer<T>, Thread> producers = new ConcurrentDictionary<IProducer<T>, Thread>();
        private readonly ConcurrentQueue<T> data = new ConcurrentQueue<T>();
        private readonly Mutex workMutex = new Mutex();
        private readonly SemaphoreSlim productionSemaphore = new SemaphoreSlim(0, Int32.MaxValue);
        private Boolean stopped = false;

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

            while (producers.Count != 0);

            while (productionSemaphore.CurrentCount != 0);

            Int32 waitCount = consumers.Count;
            for (int i = 0; i < waitCount; i++) {
                productionSemaphore.Release();
            }

            while (consumers.Count != 0);
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

                Thread thread;
                consumers.TryRemove(consumer, out thread);
            });

            consumers[consumer] = thread;
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

                Thread thread;
                producers.TryRemove(producer, out thread);
            });

            producers[producer] = thread;
        }
    }
}
