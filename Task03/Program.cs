using System;

namespace Task03
{
    class Program
    {
        static void Main(string[] args)
        {
            Producer<int>[] producers = new Producer<int>[GeneralResources.amountProducers];
            Consumer<int>[] consumers = new Consumer<int>[GeneralResources.amountConsumers];

            for (int i = 0; i < GeneralResources.amountProducers; i++)
            {
                producers[i] = new Producer<int>($"Producer {i + 1}", ref GeneralResources.mainBuffer);
                producers[i].Start();
            }
            for (int i = 0; i < GeneralResources.amountConsumers; i++)
            {
                consumers[i] = new Consumer<int>($"Consumer {i + 1}", ref GeneralResources.mainBuffer);
                consumers[i].Start();
            }

            Console.ReadKey();

            for (int i = 0; i < GeneralResources.amountProducers; i++)
                producers[i].Stop();
            for (int i = 0; i < GeneralResources.amountConsumers; i++)
                consumers[i].Stop();
        }
    }
}
