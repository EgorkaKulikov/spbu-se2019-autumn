using System;
using System.Collections.Generic;
using System.Threading;

namespace Task03
{
    internal static class Program
    {
        public static void Main(string[] args)
        {
            var consWorkers = new List<Consumer<int>>();
            var prodWorkers = new List<Producer<int>>();

            for (var num = 0; num < Constants.ConsumersCnt; num++)
            {
                var consumer = new Consumer<int>();
                consWorkers.Add(consumer);
                var consThread = new Thread(consumer.ReadData);
                consThread.Start();
            }

            for (var num = 0; num < Constants.ProducersCnt; num++)
            {
                var producer = new Producer<int>();
                prodWorkers.Add(producer);
                var prodThread = new Thread(producer.WriteData);
                prodThread.Start();
            }

            //Waiting for key press
            Console.ReadKey();

            //End reading and writing
            foreach (var consumer in consWorkers) consumer.StopRunning();
            foreach (var producer in prodWorkers) producer.StopRunning();
        }
    }
}