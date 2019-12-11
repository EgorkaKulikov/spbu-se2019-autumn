using System.Collections.Generic;
using System.Threading;

namespace Task03
{
    public class Shared<T>
    {
        public static readonly List<T> List = new List<T>();
        
        public static readonly Semaphore full = new Semaphore(0, int.MaxValue);
        public static readonly Mutex mutexProd = new Mutex();
        public static readonly Mutex mutexCons = new Mutex();
    }
}