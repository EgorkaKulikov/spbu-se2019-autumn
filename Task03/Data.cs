using System.Collections.Generic;
using System.Threading;

namespace Task03
{
    public static class Data<T>
    {
        public static readonly Queue<T> Buffer = new Queue<T>();
        public static int ReadCnt = 0;
        public static int WriteCnt = 0;
        //Primitives for producer-consumer algorithm
        public static Semaphore BufSemaphore = new Semaphore(0, Constants.SemaphoreMaxCnt);
        public static Mutex BufMutex = new Mutex();
    }
}