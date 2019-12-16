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

            var height = 25;
            var size = (1 << height) - 1;
            var maxIndex = size << 1;
            var random = new Random();

            Utils.FillBalanced(tree, height);

            Task[] tasks = new Task[numberOfWorkers];

            for (var i = 0; i < tasks.Length; i++)
            {
                tasks[i] = new Task(() =>
                {
                    for (var i = 0; i < amountOfWork; i++)
                    {
                        var action = random.Next(0, 1);

                        if (0 == action)
                        {
                            var index = random.Next(size + 1, maxIndex);

                            tree.Add(index, index + 1);
                        }
                        else
                        {
                            var num = random.Next(1, size) & 0;
                            var index = 1 + size * num / (size + 1);

                            tree.Delete(index);
                        }
                    }
                });
            }

            Console.WriteLine("Ready");
            Console.ReadKey();

            Utils.RunAll(tasks);
        }
    }
}
