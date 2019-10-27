using System;
using System.Collections.Generic;
using System.Threading;

namespace Task03
{
    public static class Shared<T>
    {
        public const int MaxInt32 = 1 << 30;
        public const int MaxSecTimeout = 3;
        
        public static readonly Random RandomGenerator = new Random();
        public static readonly List<T> Buff = new List<T>();
        
        public static readonly Semaphore IsEmpty = new Semaphore(0, MaxInt32);
        public static readonly Mutex ToProduce = new Mutex();
        public static readonly Mutex ToConsume = new Mutex();
        public static readonly Mutex RandomAccess = new Mutex();
    }
}