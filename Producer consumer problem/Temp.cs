using System.Collections.Generic;
using System.Threading;

namespace Producer_consumer_problem
{
    public class Temp<T>
    {
        public static readonly Queue<T> Buffer = new Queue<T>();

        public static Semaphore Ready = new Semaphore(0, int.MaxValue);
        public static Mutex Put = new Mutex();
        public static Mutex Get = new Mutex();

        public static int Size()
        {
            return Buffer.Count;
        }
    }
}