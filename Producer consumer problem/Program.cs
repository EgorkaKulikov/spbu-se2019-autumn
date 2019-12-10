using System;
using System.Collections.Generic;

namespace Producer_consumer_problem
{
    internal class Program
    {
        private static void Main()
        {
            Console.WriteLine("Input producers number:");
            Info.ProducersCount = Convert.ToInt32(Console.ReadLine());

            Console.WriteLine("Input consumers number:");
            Info.ConsumersCount = Convert.ToInt32(Console.ReadLine());

            var producers = new List<Producer<int>>();
            var consumers = new List<Consumer<int>>();

            if (Info.ProducersCount != 0) Info.AllProducersDone = false;

            for (var i = 0; i < Info.ProducersCount; ++i) producers.Add(new Producer<int>());

            if (Info.ConsumersCount != 0) Info.AllConsumersDone = false;

            for (var i = 0; i < Info.ConsumersCount; ++i) consumers.Add(new Consumer<int>());

            Console.ReadKey();

            for (var i = 0; i < Info.ProducersCount; ++i) producers[i].Stop();

            for (var i = 0; i < Info.ConsumersCount; ++i) consumers[i].Stop();
        }
    }
}