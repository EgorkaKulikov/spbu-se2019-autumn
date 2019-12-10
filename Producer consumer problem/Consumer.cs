using System;
using System.Threading;

namespace Producer_consumer_problem
{
    public class Consumer<T> where T : new()
    {
        private readonly int _id;

        private bool _isRunning = true;

        public Consumer()
        {
            var myThread = new Thread(GetData);
            _id = myThread.ManagedThreadId;
            myThread.Start();
        }

        public void Stop()
        {
            _isRunning = false;
        }

        private void GetData()
        {
            Info.Print("Consumer", _id, Temp<T>.Size(), "started");

            while (_isRunning && !Info.AllProducersDone)
            {
                Temp<T>.Ready.WaitOne();
                Temp<T>.Get.WaitOne();

                Temp<T>.Buffer.Dequeue();

                Info.Print("Consumer", _id, Temp<T>.Size(), "removing");
                Thread.Sleep((int) TimeSpan.FromSeconds(Info.WaitTimeInSeconds).TotalMilliseconds);

                Temp<T>.Get.ReleaseMutex();
            }

            if (Interlocked.Decrement(ref Info.AliveConsumers) == 0) Info.AllConsumersDone = true;

            Info.Print("Consumer", _id, Temp<T>.Size(), "ended");
        }
    }
}