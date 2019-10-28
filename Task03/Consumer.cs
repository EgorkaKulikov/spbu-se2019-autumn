using System.Threading;

namespace Task03
{
    class Consumer<T>
    {
        private volatile bool cancelRequired = false;

        public Consumer()
        {
            var t = new Thread(Work);
            t.Start();
        }

        private void Work()
        {
            while (!cancelRequired)
            {
                Shared<T>.IsEmpty.WaitOne();

                Shared<T>.ConsumeMutex.WaitOne();
                var data = Shared<T>.Data.Dequeue();
                Shared<T>.ConsumeMutex.ReleaseMutex();

                Shared<T>.SharedRandomAccess.WaitOne();
                var sleepTimeout = Shared<T>.SharedRandom.Next(Default.MinTimeout, Default.MaxTimeout);
                Shared<T>.SharedRandomAccess.ReleaseMutex();

                Thread.Sleep(sleepTimeout);
            }
        }

        public void Stop() => cancelRequired = true;
    }
}
