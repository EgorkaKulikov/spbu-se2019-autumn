using System;
using System.Collections.Generic;
using System.Threading;

namespace ConsoleApplication1
{
    class Producer
    {
        private static Mutex _mp = new Mutex();
        private static int twoReadings = 0;
        internal void Set(object data)
        {
            _mp.WaitOne();
            Program.Buf.Add(data);
            if (twoReadings % 2 == 1)
            {
                Thread.Sleep(1000);
                _mp.ReleaseMutex();
                Program.Sem.Release(2);
            }
            else
            {
                _mp.ReleaseMutex();
                Interlocked.Increment(ref twoReadings);
            }
        }
    }

    class Consumer
    {
        private static Mutex _mc = new Mutex();
        internal void Get(object number) 
        {
            Program.Sem.WaitOne();
            _mc.WaitOne();
            Thread.Sleep(1);
            if (Program.Buf.Count != 0)
            {
                object data = Program.Buf[0];
                Program.Buf.RemoveAt(0);
                Console.WriteLine("Consumer" + number + " took " + data);
            }
            else Program.Sem.Release();
            _mc.ReleaseMutex();
        }
    }
    
    internal class Program
    {
        internal static List<object> Buf = new List<object>();
        internal static Semaphore Sem = new Semaphore(0, Int32.MaxValue);

        public static void Main(string[] args)
        {
            if (args.Length < 2) Console.WriteLine("You did not provide a number of producers or/and consumers.");
            else
            {
                int prod = Convert.ToInt32 (args[0]);
                int cons = Convert.ToInt32 (args[1]);
                Thread[] threads = new Thread[prod + cons];
                for (int i = 0; i < prod; i++)
                {
                    Producer p = new Producer();
                    threads[i] = new Thread(data =>
                    {
                        p.Set(data);
                    });
                    threads[i].Start(i);
                }
                for (int i = prod; i < cons + prod; i++) 
                { 
                    Consumer c = new Consumer();
                    threads[i] = new Thread(number =>
                    {
                        c.Get(number);
                    });
                    threads[i].Start(i - prod + 1);
                }
                
                ConsoleKeyInfo cki = new ConsoleKeyInfo();
                int finish = 0;
                while (finish == 0)
                {
                    cki = Console.ReadKey(true);
                    if (cki.KeyChar < 255 && cki.KeyChar > 0)
                    {
                        finish = 1;
                        for (int i = 0; i < prod + cons; i++)
                        {
                            threads[i].Join();
                        }
                    }
                }
            }
        }
    }
}