using System;
using System.Collections.Generic;
using System.Threading;

namespace Task03 {
    public abstract class SomeWorker: IWorker {
        protected static Int32 unhandled = 0;

        public static Int32 unhandledCount {
            get {
                return unhandled;
            }
        }

        protected Random random = new Random();
        protected Int32 id;
        protected Int32 workAmount = 0;
        public Boolean isActive {
            get {
                return true;
            }
        }

        protected SomeWorker(Int32 id) {
            this.id = id;
        }
    }

    public class SomeConsumer: SomeWorker, IConsumer<String> {
        public SomeConsumer(Int32 id): base(id) {}

        public void Consume(String data) {
            workAmount++;
            Interlocked.Decrement(ref unhandled);
            Console.WriteLine($"Consumer {id}: {workAmount} - {data}");

            if (workAmount % 2 == 0) {
                Thread.Sleep(random.Next(100, 500));
            }
        }
    }

    public class SomeProducer: SomeWorker, IProducer<String> {
        public SomeProducer(Int32 id): base(id) {}

        public IEnumerable<String> Produce() {
            Thread.Sleep(random.Next(100, 500));

            for (int i = 0; i < 2; i++) {
                workAmount++;
                Interlocked.Increment(ref unhandled);
                yield return $"{workAmount} line produced by {id}";
            }

            yield break;
        }
    }
}
