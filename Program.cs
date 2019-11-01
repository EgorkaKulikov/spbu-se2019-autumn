using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;

namespace Task03
{
    class Program
    {
        static void Main(string[] args)
        {
            Actions.PrintInfo("Program", 0, "is about to get started!");

            var producers = new List<Producer<int>>();
            var consumers = new List<Consumer<int>>();

            for (int i = 0; i < Config.numProducers; i++)
            {
                producers.Add(new Producer<int>(i));
            }
            for (int i = 0; i < Config.numConsumers; i++)
            {
                consumers.Add(new Consumer<int>(i));
            }

            Console.ReadKey();
            Actions.PrintInfo("Program", 0, "is requested to be closed");

            for (int i = 0; i < Config.numProducers; i++)
            {
                producers[i].Cancel();
            }
            for (int i = 0; i < Config.numConsumers; i++)
            {
                consumers[i].Cancel();
            }
            Data<int>.full.Release(Config.numConsumers);

            Console.ReadLine();
        }
    }

    public class Data<T>
    {
        public static Queue<T> buffer = new Queue<T>();

        public static Mutex mProducer = new Mutex();
        public static Mutex mConsumer = new Mutex();
        public static Semaphore full = new Semaphore(0, int.MaxValue);
    }

    public static class Config
    {
        public const int minWaitTime = 1000;
        public const int maxWaitTime = 3000;

        public const int numProducers = 3;
        public const int numConsumers = 5;
    }

    public static class Actions
    {
        private static Random random = new Random();

        public static int getSleepTime()
        {
            return random.Next(Config.minWaitTime, Config.maxWaitTime);
        }

        public static void PrintInfo(string name, int id, string action, int curSize)
        {
            Console.WriteLine(name + "#" + id + " " + action + ", cur size:" + curSize);
        }
        public static void PrintInfo(string name, int id, string action)
        {
            Console.WriteLine(name + "#" + id + " " + action);
        }
    }
}
