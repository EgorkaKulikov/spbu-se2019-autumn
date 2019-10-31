using System;
using System.Collections.Generic;
using System.Threading;

namespace Task03
{
    public class SharedRes<T>
    {
        public static readonly List<T> Buffer = new List<T>();
        public static Semaphore FullBuff = new Semaphore(0, It.Max);

        public static readonly Mutex MProd = new Mutex();
        public static readonly Mutex MCons = new Mutex();
    }
}