using System;
using System.Threading;

namespace Task03
{
    class Producer<T> where T : new()
    {
        string name;
        Buffer<T> buffer;
        bool canContinue;

        public Producer(string name, ref Buffer<T> buffer)
        {
            this.name = name;
            this.buffer = buffer;
            canContinue = true;
        }

        public void Start()
        {
            Thread thread = new Thread(() =>
            {
                GeneralResources.amountWorkingProducers++;
                Console.WriteLine($"{name} begin do some work.");
                while (canContinue)
                {
                    Console.WriteLine($"{name} want to put data in buffer.");
                    PutData();
                    Console.WriteLine($"{name} put data in buffer.");
                }
                GeneralResources.amountWorkingProducers--;
                Console.WriteLine($"{name} end do some work.");
            });
            thread.Start();
        }

        public void Stop()
        {
            canContinue = false;
        }

        T SomeWork()
        {
            Thread.Sleep(GeneralResources.random.Next(GeneralResources.workTimeProducer.Item1, GeneralResources.workTimeProducer.Item2));
            return new T();
        }

        void PutData()
        {
            T resultSomeWork = SomeWork();
            buffer.Push(resultSomeWork);
        }
    }
}
