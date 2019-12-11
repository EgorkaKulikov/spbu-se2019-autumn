using System;
using System.Collections.Generic;

namespace Consumers
{
    public static class Program
    {
        public static void Main()
        {
            List<Producer<int>> producers = new List<Producer<int>>();
            List<Consumer<int>> consumers = new List<Consumer<int>>();
            
            for (var i = 0; i < It.ProducersNum; i++)
                producers.Add(new Producer<int>(i));

            for (var i = 0; i < It.ConsumersNum; i++)
                consumers.Add(new Consumer<int>(i));

            Console.ReadKey();

            foreach (var producer in producers)
                producer.Cancel();
            
            foreach (var consumer in consumers)
                consumer.Cancel();
        }
    }
}