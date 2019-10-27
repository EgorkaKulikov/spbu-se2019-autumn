using System;
using System.Threading;

namespace Task03
{
    public class Producer<T> where T : new()
    {
        private readonly int _id;
        private volatile bool _cancelled = false;

        public Producer(int newId)
        {
            _id = newId;
            Console.WriteLine("$Producer #{_id} has joined the session");
            var thread = new Thread(Produce);
            thread.Start();
        }

        public void Produce()
        {
            while (!_cancelled)
            {
                Shared<T>.ToProduce.WaitOne();
                Shared<T>.Buff.Add(new T());
                Shared<T>.IsEmpty.Release(1);
                Shared<T>.ToProduce.ReleaseMutex();
                Console.WriteLine($"Producer #{_id} has produced some data");
                Shared<T>.RandomAccess.WaitOne();
                var timeout = Shared<T>.RandomGenerator.Next(1, Shared<T>.MaxSecTimeout) * 1000;
                Shared<T>.RandomAccess.ReleaseMutex();
                Thread.Sleep(timeout);
            }
            
            Console.WriteLine($"Producer #{_id} is dead. Rest in pieces");
        }

        public void Cancel()
        {
            _cancelled = true;
            Console.WriteLine($"Producer #{_id} has done goofed");
        }
    }
}