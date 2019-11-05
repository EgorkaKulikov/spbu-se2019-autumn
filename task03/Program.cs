using System;
using System.Linq;
using System.Collections.Generic;
using System.Threading;

namespace Task03 {
    class Program {
        private class Settings {
            public static int consumersCount = 5;
            public static int producersCount = 5;
            public static IAccessPolicy policy = new SomePolicy();
        }

        private static void Main() {
            var storage = new Storage<String>(Settings.policy);

            var workers = new List<SomeWorker>();

            foreach (int i in Enumerable.Range(1, Settings.consumersCount)) {
                workers.Add(new SomeConsumer(i, storage));
            }

            foreach (int i in Enumerable.Range(1, Settings.producersCount)) {
                workers.Add(new SomeProducer(i, storage));
            }

            var threads = new List<Thread>();

            Console.ReadKey();

            foreach (var worker in workers) {
                threads.Add(new Thread(worker.DoWorking));
            }

            foreach (var thread in threads) {
                thread.Start();
            }

            Console.ReadKey();

            foreach (var worker in workers) {
                worker.Stop();
            }

            foreach (var thread in threads) {
                thread.Join();
            }

            Console.WriteLine("~)-)~");
        }
    }

    public class SomePolicy: IAccessPolicy {
        private int readersCount;
        private readonly Mutex globalMutex = new Mutex();
        private readonly Mutex countMutex = new Mutex();
        private readonly Semaphore writeMutex = new Semaphore(1, 1);

        public void AcquireRead() {
            globalMutex.WaitOne();
            countMutex.WaitOne();
            if (readersCount == 0) {
                writeMutex.WaitOne();
            }
            readersCount++;
            countMutex.ReleaseMutex();
            globalMutex.ReleaseMutex();
        }

        public void ReleaseRead() {
            countMutex.WaitOne();
            readersCount--;
            if (readersCount == 0) {
                writeMutex.Release();
            }
            countMutex.ReleaseMutex();
        }

        public void AcquireWrite() {
            globalMutex.WaitOne();
            writeMutex.WaitOne();
        }

        public void ReleaseWrite() {
            writeMutex.Release();
            globalMutex.ReleaseMutex();
        }
    }
}
