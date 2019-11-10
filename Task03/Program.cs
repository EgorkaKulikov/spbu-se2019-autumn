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
            Program.Empty.WaitOne();
            _mp.WaitOne();
            Program.Buf.Add(data);
            _mp.ReleaseMutex();
            if (twoReadings == 1)
            {
                Thread.Sleep(1000);
                Program.Full.Release(2);
                twoReadings = 0;
            }
            else twoReadings = 1;
        }
    }

    class Consumer
    {
        private static Mutex _mc = new Mutex();
        internal void Get(object number) 
        {
            Program.Full.WaitOne();
            _mc.WaitOne();
            if (Program.Buf.Count != 0)
            {
                object data = Program.Buf[0];
                Program.Buf.RemoveAt(0);
                Console.WriteLine("Consumer" + number + " took " + data);
            }
            Program.Empty.Release();
            _mc.ReleaseMutex();
        }
    }
    
    internal class Program
    {
        internal static List<object> Buf = new List<object>();
        internal static Semaphore Full;
        internal static Semaphore Empty;

        public static void Main(string[] args)
        {
            if (args.Length < 2) Console.WriteLine("You did not provide a number of producers or/and consumers.");
            else
            {
                int prod = Convert.ToInt32 (args[0]);
                int cons = Convert.ToInt32 (args[1]);
                Empty = new Semaphore(0, prod);
                Full = new Semaphore(0, cons);
                Thread[] threads = new Thread[prod + cons];
                int completed = 0;
                for (int i = 0; i < prod; i++)
                {
                    Producer p = new Producer();
                    threads[i] = new Thread(data =>
                    {
                        p.Set(data);
                        Interlocked.Increment(ref completed);
                    });
                    threads[i].Start(i);
                }
                Empty.Release(2);
                for (int i = prod; i < cons + prod; i++) 
                { 
                    Consumer c = new Consumer();
                    threads[i] = new Thread(number =>
                    {
                        c.Get(number);
                        if (Interlocked.Increment(ref completed) >= cons && prod < cons)
                        {
                            Empty.WaitOne();
                            Full.Release();
                        }
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
                            threads[i].Abort();
                        }
                    }
                }
            }
        }
    }
}