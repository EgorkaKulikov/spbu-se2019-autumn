using System;
using System.Threading;
using System.Collections.Generic;

namespace Task03 {
    public abstract class SomeWorker {
        static public bool need_to_start = false;
        static public bool need_to_stop = false;

        protected int work_amount = 0;
        protected Storage<List<String>> storage;
        protected int id;

        protected SomeWorker(int id, Storage<List<String>> storage) {
            this.id = id;
            this.storage = storage;
        }

        public abstract void doSomething();

        public void wait() {
            while (!need_to_start) {}
            doSomething();
        }
    }

    public class SomeReader: SomeWorker, Storage<List<String>>.Reader {

        public SomeReader(int id, Storage<List<String>> storage): base(id, storage) {}

        public override void doSomething() {
            while (!need_to_stop) {
                storage.add_reader(this);
                Thread.Sleep(1000);
            }
        }

        private void print(string s) {
            Console.WriteLine($"From reader {id}: {s}");
        }

        public void doReading(List<String> data) {
            print("Time has come");

            for (int i = 0; i < 2; i++) {
                if (work_amount >= data.Count) {
                    print($"There is nothing to print");
                    return;
                }

                print($"{data[work_amount]}");
                work_amount++;
            }
        }
    }

    public class SomeWriter: SomeWorker, Storage<List<String>>.Writer {

        public SomeWriter(int id, Storage<List<String>> storage): base(id, storage) {}

        public override void doSomething() {
            while (!need_to_stop) {
                storage.add_writer(this);
                Thread.Sleep(1000);
            }
        }

        public void doWriting(List<String> data) {
            for (int i = 0; i < 2; i++) {
                work_amount++;
                data.Add($"This is {work_amount} line produced by writer {id}");
            }
        }
    }
}
