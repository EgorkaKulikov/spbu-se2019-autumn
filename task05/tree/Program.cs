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
            var amountOfWork = Int32.Parse(args[1]);
            var numberOfWorkers = 8;
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

            Utils.RunAll(tasks);
        }
    }
}
