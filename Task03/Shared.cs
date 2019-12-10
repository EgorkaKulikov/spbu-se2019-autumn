using System;
using System.Collections.Generic;
using System.Threading;

namespace Task03
{
    class Shared<T>
    {
        public static readonly Queue<T> Data       = new Queue<T>();
        public static readonly Semaphore IsEmpty   = new Semaphore(0, int.MaxValue);
        public static readonly Mutex ProduceMutex  = new Mutex();
        public static readonly Mutex ConsumeMutex  = new Mutex();

        public static readonly Random SharedRandom = new Random();
        public static readonly Mutex SharedRandomAccess  = new Mutex();
    }
}
