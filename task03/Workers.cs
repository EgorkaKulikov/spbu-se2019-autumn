using System;
using System.Collections.Generic;
using System.Threading;

namespace Task03 {
    public abstract class SomeWorker {
        protected bool needToStop = false;
        protected int id;
        protected int workAmount = 0;
        protected Storage<String> storage;

        protected SomeWorker(int id, Storage<String> storage) {
            this.id = id;
            this.storage = storage;
        }

        public abstract void DoWorking();

        public void Stop() {
            needToStop = true;
        }
    }

    public class SomeConsumer: SomeWorker, Storage<String>.IConsumer {
        public SomeConsumer(int id, Storage<String> storage): base(id, storage) {}

        public override void DoWorking() {
            while (!needToStop) {
                storage.AddConsumer(this);
                Thread.Sleep(1000);
            }
        }

        private void Print(string s) {
            Console.WriteLine($"From consumer {id}: {s}");
        }

        public void DoConsuming(ListReader<String> data) {
            for (int i = 0; i < 2; i++) {
                if (workAmount >= data.Count) {
                    Print($"There is nothing to Print");
                    return;
                }

                Print($"{data.Read(workAmount)}");
                workAmount++;
            }
        }
    }

    public class SomeProducer: SomeWorker, Storage<String>.IProducer {
        public SomeProducer(int id, Storage<String> storage): base(id, storage) {}

        public override void DoWorking() {
            while (!needToStop) {
                storage.AddProducer(this);
                Thread.Sleep(1000);
            }
        }

        public void DoProducing(List<String> data) {
            for (int i = 0; i < 2; i++) {
                workAmount++;
                data.Add($"This is {workAmount} line produced by {id}");
            }
        }
    }
}
