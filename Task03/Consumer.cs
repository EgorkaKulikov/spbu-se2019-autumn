using System;
using System.Threading;

namespace Task03
{
    class Consumer<T> where T : new()
    {
        string name;
        Buffer<T> buffer;
        bool canContinue;

        public Consumer(string name, ref Buffer<T> buffer)
        {
            this.name = name;
            this.buffer = buffer;
            canContinue = true;
        }

        public void Start()
        {
            Thread thread = new Thread(() =>
            {
                bool gotDataFromBuffer = true;
                Console.WriteLine($"{name} begin do some work.");
                while (canContinue || !buffer.IsEmpty() || GeneralResources.amountWorkingProducers > 0)
                {
                    if (gotDataFromBuffer) Console.WriteLine($"{name} want to get data from buffer.");
                    gotDataFromBuffer = GetData();
                    if (gotDataFromBuffer) Console.WriteLine($"{name} got data from buffer.");
                }
                Console.WriteLine($"{name} end do some work.");
            });
            thread.Start();
        }

        public void Stop()
        {
            canContinue = false;
        }

        void SomeWork(T data)
        {
            Thread.Sleep(GeneralResources.random.Next(GeneralResources.workTimeConsumer.Item1, GeneralResources.workTimeConsumer.Item2));
        }

        bool GetData()
        {
            Maybe<T> newData = buffer.Pop();
            if (newData != Maybe<T>.Nothing)
            {
                SomeWork(newData.getValue());
                return true;
            }
            else
            {
                Thread.Sleep(GeneralResources.waitingTimeConsumer);
                return false;
            }
        }
    }
}
