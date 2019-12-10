using System;
using System.Collections.Generic;
using System.Threading;

namespace Task03
{
    public static class SharedRes<T>
    {
        private static readonly Random _random = new Random();
        
        public static readonly List<T> Data = new List<T>();

        public static int amountConsumers = 80;
        public static int amountProducers = 80;

        public static readonly int TimeoutProducer = _random.Next(500, 1000);
        public static readonly int TimeoutConsumer = _random.Next(500, 1000);

        public static readonly Semaphore Empty = new Semaphore(0, Int32.MaxValue);
        public static readonly Mutex MConsumer = new Mutex();
        public static readonly Mutex MProducer = new Mutex();
    }
}