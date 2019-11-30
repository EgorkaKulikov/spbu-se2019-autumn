using System;
using System.Collections.Generic;
using System.Threading;

namespace Task03
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("The session has begun. Everyone, get in here!");
            List<Consumer<int>> consumers = new List<Consumer<int>>();
            List<Producer<int>> producers = new List<Producer<int>>();

            var numProducers = 7;
            var numConsumers = 7;

            for (int i = 0; i < numProducers; i++)
            {
                producers.Add(new Producer<int>(i));
            }
            for (int i = 0; i < numConsumers; i++)
            {
                consumers.Add(new Consumer<int>(i));
            }

            while (!Console.KeyAvailable) {}
            
            Console.WriteLine("The session is closing.");
            
            for (int i = 0; i < numProducers; i++)
            {
                producers[i].Cancel();
            }

            var isEmpty = false;
            
            while (!isEmpty)
            {
                Shared<int>.ToConsume.WaitOne();
                Shared<int>.ToProduce.WaitOne();
                if (Shared<int>.Buff.Count == 0)
                {
                    isEmpty = true;
                }
                
                Shared<int>.ToConsume.ReleaseMutex();
                Shared<int>.ToProduce.ReleaseMutex();

                if (!isEmpty)
                {
                    Thread.Sleep(Shared<int>.MaxSecTimeout);
                }
                
            }
            
            Console.WriteLine("Buffer cleared.");

            for (int i = 0; i < numConsumers; i++)
            {
                consumers[i].Cancel();
            }
            
            Shared<int>.isNonEmpty.Release(numConsumers);
        }
    }
}
