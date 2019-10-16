using System;
using System.Collections.Generic;
using System.Threading;

namespace ConsoleApplication1
{
    class Producer
    {
        private static Mutex _mp = new Mutex();
        private static int Flag = 0;
        internal void Set(object data)
        {
            Program.Empty.WaitOne();
            _mp.WaitOne();
            Program.Buf.Add(data);
            _mp.ReleaseMutex();
            if (Flag == 1)
            {
                Thread.Sleep(1000);
                Program.Full.Release(2);
                Flag = 0;
            }
            else Flag = 1;
            if (Program.fin == 1) Thread.CurrentThread.Abort();
        }
    }

    class Consumer
    {
        private static Mutex _mc = new Mutex();
        internal void Get(object number) 
        {
            Program.Full.WaitOne();
            _mc.WaitOne();
            object data = Program.Buf[0];
            Program.Buf.RemoveAt(0);
            Program.Empty.Release();
            _mc.ReleaseMutex();
            if (Program.fin == 1) Thread.CurrentThread.Abort();
            Console.WriteLine("Consumer" + number + " took " + data);
        }
    }
    
    internal class Program
    {
        internal static List<object> Buf = new List<object>();
        internal static Semaphore Full;
        internal static Semaphore Empty;
        internal static int fin = 0; 

        private static void Keystroke()
        {
            ConsoleKeyInfo cki = new ConsoleKeyInfo();
            while (true)
            {
                cki = Console.ReadKey(true);
                if (cki.KeyChar < 255 && cki.KeyChar > 0)
                {
                    fin = 1;
                    break;
                }
            }
        }

        public static void Main(string[] args)
        {
            if (args.Length < 2) Console.WriteLine("You did not provide a number of producers or/and consumers.");
            else
            {
                int prod = Convert.ToInt32 (args[0]);
                int cons = Convert.ToInt32 (args[1]);
                Empty = new Semaphore(0, prod);
                Full = new Semaphore(0, cons);
                Thread keystroke = new Thread(new ThreadStart(Keystroke));
                keystroke.Start();
                for (int i = 0; i < prod; i++)
                {
                    Producer p = new Producer();
                    Thread producer = new Thread(new ParameterizedThreadStart(p.Set));
                    producer.Start(i);
                }
                for (int i = 0; i < (cons <= prod ? cons : prod); i++) 
                { 
                    Consumer c = new Consumer();
                    Thread consumer = new Thread(new ParameterizedThreadStart(c.Get));
                    consumer.Start(i + 1);
                }
                Empty.Release(2);
            }
        }
    }
}