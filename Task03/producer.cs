using System;
using System.Threading;

namespace ConsoleApp1
{
    class Producer
    {
        public Producer(string thread_name)
        {
            Thread thread = new Thread(this.produce);
            thread.Name = thread_name + " producer";
            thread.Start();
        }

        void produce()
        {
            while (!Data.end_prog)
            {
                Data.may_change.WaitOne();
                put_item();
                Data.may_change.Release();
                Data.full.Release();
                Thread.Sleep(Data.sleep_time);
            }
        }

        void put_item()
        {
            int tmp = Data.rnd.Next(0, 1000);
            Data.buff.Add(tmp);
            Console.WriteLine("thread " + Thread.CurrentThread.Name + " puted " + tmp);
        }
    }
}