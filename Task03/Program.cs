using System;
using System.Collections.Generic;
using System.Threading;

namespace ProducerConsumer
{
    internal static class Program
    {
        private static void Main()
        {
            var consumers = new List<Consumer<int>>();
            var producers = new List<Producer<int>>();

            const int amountConsumers = 6;
            const int amountProducers = 3;

            for (var i = 0; i < amountProducers; ++i)
            {
                var producer = new Producer<int>();
                producers.Add(producer);
                var thread = new Thread(producer.ProduceData);
                thread.Start();
            }

            for (var i = 0; i < amountConsumers; ++i)
            {
                var consumer = new Consumer<int>();
                consumers.Add(consumer);
                var thread = new Thread(consumer.ConsumeData);
                thread.Start();
            }

            Console.ReadKey();

            foreach (var producer in producers)
            {
                producer.StopRunning();
            }
            
            foreach (var consumer in consumers)
            {
                consumer.StopRunning();
            }

            Data<int>.Empty.Release(amountConsumers);
        }
    }
}
