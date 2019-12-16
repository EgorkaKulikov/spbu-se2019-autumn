using System;
using System.Threading.Tasks;
using System.Linq;

namespace Task05
{
    class Program
    {
        static void Main(String[] args)
        {
            var type = args[0];
            var numberOfWorkers = 8;
            var amountOfWork = 1000000;
            ITree<Int32, Int32> tree;

            if (type == "coarse")
            {
                tree = new CoarseTree<Int32, Int32>();
            }
            else
            {
                tree = new FineTree<Int32, Int32>();
            }

            var height = 26;
            var size = (1 << height) - 1;
            var fillSize = (1 << (height - 1)) - 1;
            var random = new Random();

            Utils.FillBalanced(tree, height, height - 1);

            Task[] tasks = new Task[numberOfWorkers];

            for (var i = 0; i < tasks.Length; i++)
            {
                tasks[i] = new Task(() =>
                {
                    for (var i = 0; i < amountOfWork; i++)
                    {
                        var num = random.Next(1, size) & 0;
                        var index = 1 + size * (num + 1) / (size + 1);

                        tree.Add(index, index + 1);
                    }
                });
            }

            Console.WriteLine("Ready");
            Console.ReadKey();

            Utils.RunAll(tasks);
        }
    }
}
