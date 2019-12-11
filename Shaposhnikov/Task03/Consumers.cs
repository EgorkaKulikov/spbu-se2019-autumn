using System;
using System.Threading;

namespace Consumers
{
    public class Consumer<T>
    {
        private readonly int _id;
        private  bool _isReading = true;
        private  bool _fStop = false;
        private  readonly Random _rand = new Random();

        public void Cancel()
        {
            _fStop = true;
            Console.WriteLine("Consumer {0} is marked stopping", _id);
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
                SharedRes<T>.MCons.WaitOne();

                if (SharedRes<T>.Buffer.Count == 0 && _fStop)
                {
                    _isReading = false;
                    SharedRes<T>.MCons.ReleaseMutex();
                    Console.WriteLine($"Consumer {_id} finished with zero buffer");
                    break;
                }
                
                SharedRes<T>.FullBuff.WaitOne();

                var data = SharedRes<T>.Buffer[0];

                Console.WriteLine("Consumer {0} has consumed some data, remaining buffer: {1}", _id,
                    SharedRes<T>.Buffer.Count - 1);
                
                SharedRes<T>.Buffer.RemoveAt(0);
                SharedRes<T>.MCons.ReleaseMutex();
                
                //manipulations with data
                
                Thread.Sleep(_rand.Next(1, It.TimeOut));
            }
        }
    }
}