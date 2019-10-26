using System.Collections.Generic;
using System.Threading;

namespace Task03
{
    public static class Data<T>
    {
        public static readonly Queue<T> buffer = new Queue<T>();

        //Primitives for producer-consumer algorithm
        public static Semaphore fullSemaphore = new Semaphore(0, Constants.SemaphoreMaxCnt);
        public static Semaphore emptySemaphore = new Semaphore(Constants.SemaphoreMaxCnt, Constants.SemaphoreMaxCnt);
        public static Mutex mutex = new Mutex();
    }
}