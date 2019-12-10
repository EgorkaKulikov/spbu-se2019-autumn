using System;
using System.Threading;

namespace Task03
{
    class Program
    {
        static void CreateProducers()
        {
            Random random = new Random();
            
            for (int currProd = 0; currProd < SharedRes<int>.amountProducers; ++currProd)
            {
                Producer<int> producer = new Producer<int>();
                Thread thr = new Thread(() => producer.PutData(random.Next(1, 101)));
                thr.Start();
            }
        }

        static void CreateConsumers()
        {
            for (int currCons = 0; currCons < SharedRes<int>.amountConsumers; ++currCons)
            {
                Consumer<int> consumer = new Consumer<int>();
                Thread thr = new Thread(() => consumer.GetData());
                thr.Start();
            }
        }

        static void Main()
        {
            CreateProducers();
            CreateConsumers();
            
            Console.ReadKey();

            Producer<int>.StopInserting();
            Consumer<int>.StopGetting();

            for (int i = 0; i < SharedRes<int>.amountConsumers; ++i)
            {
                SharedRes<int>.Empty.Release();
            }
        }
    }
}