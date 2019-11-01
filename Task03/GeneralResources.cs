using System;

namespace Task03
{
    class GeneralResources
    {
        public static Random random = new Random();
        public static int waitingTimeConsumer = (int)(workTimeProducer.Item2 * 0.6 + workTimeProducer.Item1 * 0.4);
        public static (int, int) workTimeProducer = (1000, 6000);
        public static (int, int) workTimeConsumer = (200, 1200);
        public static int amountConsumers = 6;
        public static int amountProducers = 2;
        public static Buffer<object> mainBuffer = new Buffer<object>();
    }
}
