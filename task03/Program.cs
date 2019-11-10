using System;
using System.Linq;

namespace Task03
{
    class Program
    {
        private class Settings
        {
            public static Int32 consumersCount = 5;
            public static Int32 producersCount = 50;
        }

        private static void Main()
        {
            var storage = new Storage<String>();

            foreach (Int32 i in Enumerable.Range(1, Settings.consumersCount))
            {
                storage.AddConsumer(new SomeConsumer(i));
            }

            foreach (Int32 i in Enumerable.Range(1, Settings.producersCount))
            {
                storage.AddProducer(new SomeProducer(i));
            }

            Console.ReadKey();

            storage.Start();

            Console.ReadKey();

            storage.Stop();

            Console.WriteLine($"Unhandled: {SomeWorker.UnhandledCount}");
        }
    }
}
