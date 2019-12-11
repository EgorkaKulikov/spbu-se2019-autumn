using System;
using System.Threading;

namespace Task03
{
    public class Producer<T> where T : new()
    {
        private readonly int _id;
        private  bool _isWriting = true;
        private  readonly Random _rand = new Random();

        public void Cancel()
        {
            _isWriting = false;
            Console.WriteLine("Producer {0} is stopping", _id);
        }

        public Producer(int id)
        {
            this._id = id;
            Console.WriteLine("Producer {0} is running", this._id);
            var thread = new Thread(PutData);
            thread.Start();
        }

        private void PutData()
        {
            while (_isWriting)
            {
                SharedRes<T>.MProd.WaitOne();

                SharedRes<T>.Buffer.Add(new T());

                Console.WriteLine("Producer {0} has produced some data, actual buffer: {1}", _id,
                    SharedRes<T>.Buffer.Count);
                
                SharedRes<T>.FullBuff.Release();
                SharedRes<T>.MProd.ReleaseMutex();

                Thread.Sleep(_rand.Next(1, It.TimeOut));
            }
        }
    }
}