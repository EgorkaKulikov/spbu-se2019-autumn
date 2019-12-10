using System;

namespace gen
{
    class Program
    {
        static void Main(String[] args)
        {
            var numberOfVertices = Int32.Parse(args[0]);
            var minWeight = Int32.Parse(args[1]);
            var maxWeight = Int32.Parse(args[2]);
            var freqNumerator = Int32.Parse(args[3]);
            var freqDenumerator = Int32.Parse(args[4]);
            var random = new Random();

            Console.WriteLine(numberOfVertices);

            for (Int32 i = 0; i < numberOfVertices; i++)
            {
                for (Int32 j = i + 1; j < numberOfVertices; j++)
                {
                    if (random.Next(0, freqDenumerator) < freqNumerator)
                    {
                        Console.WriteLine($"{random.Next(0, numberOfVertices)} {random.Next(0, numberOfVertices)} {random.Next(minWeight, maxWeight + 1)}");
                    }
                }
            }
        }
    }
}
