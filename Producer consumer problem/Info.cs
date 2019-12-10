using System;

namespace Producer_consumer_problem
{
    public class Info
    {
        public static int ProducersCount = 42;
        public static int ConsumersCount = 42;
        public static int WaitTimeInSeconds = 1;

        public static int AliveProducers = ProducersCount;
        public static int AliveConsumers = ConsumersCount;

        public static bool AllProducersDone = true;
        public static bool AllConsumersDone = true;

        public static void Print(string name, int id, int bufferSize, string action)
        {
            Console.WriteLine($"{name,8} (id: {id,2}) " +
                              $"{action,8} " +
                              $"(buffer size: {bufferSize,2})");
        }
    }
}