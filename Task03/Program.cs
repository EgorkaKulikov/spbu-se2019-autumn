using System;

namespace Task03
{
    class Program
    {
        static void Main(string[] args)
        {
            Consumer<object>[] consumers = new Consumer<object>[GeneralResources.amountConsumers];
            Producer<object>[] producers = new Producer<object>[GeneralResources.amountProducers];

            for (int i = 0; i < GeneralResources.amountConsumers; i++)
            {
                consumers[i] = new Consumer<object>($"Consumer {i + 1}", ref GeneralResources.mainBuffer);
                consumers[i].Start();
            }
            for (int i = 0; i < GeneralResources.amountProducers; i++)
            {
                producers[i] = new Producer<object>($"Producer {i + 1}", ref GeneralResources.mainBuffer);
                producers[i].Start();
            }

            Console.ReadKey();

            for (int i = 0; i < GeneralResources.amountConsumers; i++)
                consumers[i].Stop();
            for (int i = 0; i < GeneralResources.amountProducers; i++)
                producers[i].Stop();
        }
    }
}
