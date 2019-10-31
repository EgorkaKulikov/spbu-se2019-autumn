using System;
using System.Threading;

namespace Task03
{
    public class Consumer<T>
    {
        private readonly int _id;
        private  bool _isReading = true;
        private  readonly Random _rand = new Random();

        public void Cancel()
        {
            _isReading = false;
            Console.WriteLine("Consumer {0} is stopping", _id);
        }

        public Consumer(int id)
        {
            this._id = id;
            Console.WriteLine("Consumer {0} is running", this._id);
            var thread = new Thread(GetData);
            thread.Start();
        }

        private void GetData()
        {
            while (_isReading)
            {
                SharedRes<T>.FullBuff.WaitOne();
                SharedRes<T>.MCons.WaitOne();

                if (!_isReading)
                {
                    SharedRes<T>.MCons.ReleaseMutex();
                    SharedRes<T>.FullBuff.Release();
                    break;
                }

                var data = SharedRes<T>.Buffer[0];

                Console.WriteLine("Consumer {0} has consumed some data, remaining buffer: {1}", _id,
                    SharedRes<T>.Buffer.Count);
                
                SharedRes<T>.Buffer.RemoveAt(0);
                SharedRes<T>.MCons.ReleaseMutex();
                
                //manipulations with data
                
                Thread.Sleep(_rand.Next(1, It.TimeOut));
            }
        }
    }
}