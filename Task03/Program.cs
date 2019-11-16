using System;

namespace Task03
{
    class Program
    {
        static void Main(string[] args)
        {
            Producer<object>[] producers = new Producer<object>[GeneralResources.amountProducers];
            Consumer<object>[] consumers = new Consumer<object>[GeneralResources.amountConsumers];

            for (int i = 0; i < GeneralResources.amountProducers; i++)
            {
                producers[i] = new Producer<object>($"Producer {i + 1}", ref GeneralResources.mainBuffer);
                producers[i].Start();
            }
            for (int i = 0; i < GeneralResources.amountConsumers; i++)
            {
                consumers[i] = new Consumer<object>($"Consumer {i + 1}", ref GeneralResources.mainBuffer);
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
