using System.Collections.Generic;
using System.Threading;

namespace ProducerConsumer
{
    public static class Data<T>
    {
        public const int Timeout = 1000;
        public static readonly List<T> Buffer = new List<T>();
        public static readonly Semaphore Empty = new Semaphore(0, int.MaxValue);
        public static readonly Mutex Mutex = new Mutex();
    }
}