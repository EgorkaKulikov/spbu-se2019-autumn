using System;
using System.Threading;

namespace ConsoleApp1
{
    class Consumer
    {
        public Consumer(string thread_name)
        {
            Thread thread = new Thread(this.consume);
            thread.Name = thread_name + " consumer";
            thread.Start();
        }

        void consume()
        {
            while (!Data.end_prog)
            {
                Data.full.WaitOne();
                Data.mutex.WaitOne();
                get_item();
                Data.mutex.ReleaseMutex();
                Thread.Sleep(Data.sleep_time);
            }
            Data.cons_ended++;
        }

        void get_item()
        {
            int tmp = Data.buff[0];
            Data.buff.RemoveAt(0);
            Console.WriteLine("thread " + Thread.CurrentThread.Name + " geted " + tmp);
        }
    }
}