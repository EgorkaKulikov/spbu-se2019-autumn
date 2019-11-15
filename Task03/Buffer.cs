using System.Collections.Generic;
using System.Threading;

namespace Task03
{
    class Buffer<T> where T: new()
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

        public Maybe<T> Pop()
        {
            mutBuffer.WaitOne();
            Maybe<T> res = queue.Count > 0 ? Maybe<T>.Just(queue.Dequeue()) : Maybe<T>.Nothing;
            mutBuffer.ReleaseMutex();
            return res;
        }

        public bool IsEmpty()
        {
            return queue.Count == 0;
        }
    }
}
