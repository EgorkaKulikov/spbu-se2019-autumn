using System;
using System.Diagnostics;
using System.Threading.Tasks;
using System.Linq;

namespace Task05
{
    class Program
    {
        static void Main(String[] args)
        {
            var type = args[0];
            var numberOfWorkers = Int32.Parse(args[1]);
            var amountOfWork = 1000;
            ITree<Int32, Int32> tree;

            if (type == "coarse")
            {
                tree = new CoarseTree<Int32, Int32>();
            }
            else
            {
                tree = new FineTree<Int32, Int32>();
            }

            var distribution = Utils.GetSimpleDistribution(numberOfWorkers, amountOfWork);
            Task[] tasks;

            tasks = Utils.GetInsertionTasks(tree, distribution);

            var stopwatch = new Stopwatch();

            stopwatch.Start();
            Utils.RunAll(tasks);
            stopwatch.Stop();

            Console.WriteLine(stopwatch.ElapsedMilliseconds);
        }
    }
}
