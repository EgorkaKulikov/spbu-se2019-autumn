using System;
using System.Collections.Generic;

namespace Task03
{
    class Program
    {
        static void Main()
        {
            List<Producer<int>> producers = new List<Producer<int>>();
            List<Consumer<int>> consumers = new List<Consumer<int>>();

            for (int i = 0; i < Default.ProducersNumber; i++)
                producers.Add(new Producer<int>());

            for (int i = 0; i < Default.ConsumersNumber; i++)
                consumers.Add(new Consumer<int>());

            Console.ReadKey();

            foreach (var consumer in consumers)
                consumer.Stop();

            foreach (var producer in producers)
                producer.Stop();
        }
    }
}
