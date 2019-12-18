using System.Collections.Generic;
using System.Threading;

namespace Task03
{
    class Buffer<T> where T: struct
    {
        Mutex mutBuffer;
        Queue<T> queue;

        public Buffer()
        {
            mutBuffer = new Mutex();
            queue = new Queue<T>();
        }

        public void Push(T elem)
        {
            mutBuffer.WaitOne();
            queue.Enqueue(elem);
            mutBuffer.ReleaseMutex();
        }

        public T? Pop()
        {
            mutBuffer.WaitOne();
            T? res = queue.Count > 0 ? queue.Dequeue() : (T?)null;
            mutBuffer.ReleaseMutex();
            return res;
        }

        public bool IsEmpty()
        {
            mutBuffer.WaitOne();
            bool res = queue.Count == 0;
            mutBuffer.ReleaseMutex();
            return res;
        }
    }
}
