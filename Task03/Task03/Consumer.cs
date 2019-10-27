using System;
using System.Threading;

namespace Task03
{
    public class Consumer<T>
    {
        private readonly int _id;
        private volatile bool _cancelled = false;
        
        public Consumer(int newId)
        {
            _id = newId;
            Console.WriteLine($"Consumer #{_id} has joined the session");
            var thread = new Thread(Consume);
            thread.Start();
        }
        public void Consume()
        {
            while (!_cancelled)
            {
                Shared<T>.IsEmpty.WaitOne();

                if (_cancelled)
                {
                    break;
                }
                
                Shared<T>.ToConsume.WaitOne();
                var data = Shared<T>.Buff[0];
                Shared<T>.Buff.RemoveAt(0);
                Shared<T>.ToConsume.ReleaseMutex();
                Console.WriteLine($"Consumer #{_id} has consumed some data");
                Shared<T>.RandomAccess.WaitOne();
                var timeout = Shared<T>.RandomGenerator.Next(1, Shared<T>.MaxSecTimeout) * 1000;
                Shared<T>.RandomAccess.ReleaseMutex();
                Thread.Sleep(timeout);
                
            }
            
            Console.WriteLine($"Consumer #{_id} is dead. Rest in pieces");
        }

        public void Cancel()
        {
            _cancelled = true;
            Console.WriteLine($"Consumer #{_id} has done goofed");
        }
    }
}