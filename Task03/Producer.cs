using System.Threading;

namespace Task03
{
    class Producer<T> where T : new()
    {
        private volatile bool cancelRequired = false;

        public Producer()
        {
            var t = new Thread(Work);
            t.Start();
        }

        public void Work()
        {
            while (!cancelRequired)
            {
                Shared<T>.ProduceMutex.WaitOne();
                Shared<T>.Data.Enqueue(new T());
                Shared<T>.IsEmpty.Release(1);
                Shared<T>.ProduceMutex.ReleaseMutex();

                Shared<T>.SharedRandomAccess.WaitOne();
                var sleepTimeout = Shared<T>.SharedRandom.Next(Default.MinTimeout, Default.MaxTimeout);
                Shared<T>.SharedRandomAccess.ReleaseMutex();

                Thread.Sleep(sleepTimeout);
            }
        }

        public void Stop() => cancelRequired = true;
    }
}
