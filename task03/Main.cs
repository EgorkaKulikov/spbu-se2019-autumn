using System;
using System.Collections.Generic;
using System.Threading;

namespace Task03 {
    class Program {
        private class Settings {
            public static int readers_count = 5;
            public static int writers_count = 5;
            public static AccessPolicy policy = new SomePolicy();
        }

        static private void Main() {
            var list = new List<String>();
            var storage = new Storage<List<String>>(Settings.policy, list);

            var readers = new List<SomeReader>();
            for (int i = 0; i < Settings.readers_count; i++) {
                readers.Add(new SomeReader(i, storage));
            }

            var writers = new List<SomeWriter>();
            for (int i = 0; i < Settings.writers_count; i++) {
                writers.Add(new SomeWriter(i, storage));
            }

            var threads = new List<Thread>();

            foreach (var writer in writers) {
                threads.Add(new Thread(writer.wait));
            }

            foreach (var reader in readers) {
                threads.Add(new Thread(reader.wait));
            }

            foreach (var thread in threads) {
                thread.Start();
            }

            Console.ReadKey();

            SomeWorker.need_to_start = true;

            Console.ReadKey();

            SomeWorker.need_to_stop = true;

            foreach (var thread in threads) {
                thread.Join();
            }

            Console.WriteLine("~)-)~");
        }
    }

    public class SomePolicy: AccessPolicy {
        private int readers_count;
        private Mutex global_mutex = new Mutex();
        private Mutex count_mutex = new Mutex();
        private Semaphore write_mutex = new Semaphore(1, 1);

        public override void acquire_read() {
            global_mutex.WaitOne();
            count_mutex.WaitOne();
            if (readers_count == 0) {
                write_mutex.WaitOne();
            }
            readers_count++;
            count_mutex.ReleaseMutex();
            global_mutex.ReleaseMutex();
        }

        public override void release_read() {
            count_mutex.WaitOne();
            readers_count--;
            if (readers_count == 0) {
                write_mutex.Release();
            }
            count_mutex.ReleaseMutex();
        }

        public override void acquire_write() {
            global_mutex.WaitOne();
            write_mutex.WaitOne();
        }

        public override void release_write() {
            write_mutex.Release();
            global_mutex.ReleaseMutex();
        }
    }

}
