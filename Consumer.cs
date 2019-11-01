using System;
using System.Threading;

namespace Task03
{
    public class Consumer<T>
    {
        private readonly int id;
        private bool isCancelled;
        private string name = "Consumer";

        public Consumer(int id)
        {
            this.id = id;
            this.isCancelled = false;
            Thread myThread = new Thread(Run);
            myThread.Start();
            Actions.PrintInfo(name, id, "has been started");
        }

        public void Cancel()
        {
            this.isCancelled = true;
        }

        private void GetData()
        {
            var someData = Data<T>.buffer.Dequeue();
            Actions.PrintInfo(name, id, "read something", Data<T>.buffer.Count);
            Thread.Sleep(Actions.getSleepTime());
        }

        private void Run()
        {
            while (!isCancelled)
            {
                Data<T>.full.WaitOne();
                // in case the program is closing and we actually don't have any data
                if (isCancelled) break;
                Data<T>.mConsumer.WaitOne();
                GetData();
                Data<T>.mConsumer.ReleaseMutex();
            }

            Actions.PrintInfo(name, id, "has been stopped");
        }
    }
}
