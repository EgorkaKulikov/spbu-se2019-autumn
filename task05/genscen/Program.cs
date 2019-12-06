using System;

namespace genscen
{
    class Program
    {
        static void Main(String[] args)
        {
            var numberOfWorkers = Int32.Parse(args[0]);
            var random = new Random();

            Console.WriteLine(numberOfWorkers);

            for (var i = 0; i < numberOfWorkers; i++)
            {
                Int32 numberOfActions = random.Next(0, 1000);
                Console.WriteLine(numberOfActions);

                for (var j = 0; j < numberOfActions; j++)
                {
                    Console.WriteLine(random.Next(0, 3));
                    Console.WriteLine(random.Next(0, 1000000));
                    Console.WriteLine(random.Next(0, 1000));
                }
            }
        }
    }
}
