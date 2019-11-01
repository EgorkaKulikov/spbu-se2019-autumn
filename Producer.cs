using System;
using System.Threading;

namespace Task03
{
    public class Producer<T> where T : new()
    {
        private readonly int id;
        private bool isCancelled;
        private string name = "Producer";

        public Producer(int id)
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

        private void PutData()
        {
            Data<T>.buffer.Enqueue(new T());
            Actions.PrintInfo(name, id, "added some data", Data<T>.buffer.Count);
            Thread.Sleep(Actions.getSleepTime());
        }

        private void Run()
        {
            while (!isCancelled)
            {
                Data<T>.mProducer.WaitOne();
                PutData();
                Data<T>.full.Release(1);
                Data<T>.mProducer.ReleaseMutex();
            }

            Actions.PrintInfo(name, id, "has been finished");
        }
    }
}
