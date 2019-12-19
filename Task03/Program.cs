using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Metadata.Ecma335;
using System.Threading;
using System.Threading.Tasks;

namespace Task03
{
    public class Program
    {
        public static Mutex mutexIn = new Mutex(initiallyOwned: false);
        public static SemaphoreSlim semNotEmpty = new SemaphoreSlim(0, Int32.MaxValue);

        public static async Task Main(string[] args)
        {
            Console.WriteLine("Start");
            Data<int> data = new Data<int>();
            Task[] producers = new Task[Setup.producersNum];
            Task[] consumers = new Task[Setup.consumersNum];

            //necessary setup to cancel with a help of tokens
            var producersTokenSource = new CancellationTokenSource();
            var consumersTokenSource = new CancellationTokenSource();
            var producerToken = producersTokenSource.Token;
            var consumersToken = consumersTokenSource.Token;
            //run producers constructors
            for (int i = 0; i < Setup.producersNum; i++)
            {
                producers[i] = Task.Run(() => new Producer(producerToken), producerToken);
            }

            //run consumers constructors
            for (int i = 0; i < Setup.consumersNum; i++)
            {
                consumers[i] = Task.Run(() => new Consumer(consumersToken), consumersToken);
            }

            Console.ReadKey();
            Console.WriteLine("Cancelling");
            producersTokenSource.Cancel();

            try
            {
                await Task.WhenAll(producers);
            }
            //exception is thrown to break out of loops
            catch (OperationCanceledException)
            {
                Console.WriteLine($"\n{nameof(OperationCanceledException)} thrown, finishing\n");
            }
            finally
            {
                while (semNotEmpty.CurrentCount > 0);
                semNotEmpty.Release(Int32.MaxValue);
                consumersTokenSource.Cancel();
                try
                {
                    await Task.WhenAll(consumers);
                }
                //exception is thrown to break out of loops
                catch (OperationCanceledException)
                {
                    Console.WriteLine($"\n{nameof(OperationCanceledException)} thrown, finishing\n");
                }
                finally
                {
                    consumersTokenSource.Dispose();
                }
                producersTokenSource.Dispose();
                semNotEmpty.Dispose();
                mutexIn.Dispose();
            }
        }
    }

    class Data<T>
    {
        public Data()
        {
            list = new List<T>();
        }

        public static List<T> list;
    }
}