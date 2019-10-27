using System;
using System.Collections.Generic;

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
            for (int i = 0; i < numConsumers; i++)
            {
                consumers[i].Cancel();
            }

            Shared<int>.IsEmpty.Release(Shared<int>.MaxInt32);
        }
    }
}
