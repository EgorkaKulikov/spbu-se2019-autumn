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
            var action = args[1];
            var numberOfWorkers = 8;
            var amountOfWork = 1000;
            ITree<Int32, Int32> tree;

            switch (type)
            {
                case "coarse":
                    {
                        tree = new CoarseTree<Int32, Int32>();
                        break;
                    }
                default:
                    {
                        tree = new FineTree<Int32, Int32>();
                        break;
                    }
            }

            var distribution = Utils.GetSimpleDistribution(numberOfWorkers, amountOfWork);
            Task[] tasks;

            switch (action)
            {
                case "add":
                    {
                        tasks = Utils.GetInsertionTasks(tree, distribution);
                        break;
                    }
                case "delete":
                    {
                        Int32 height = 2 + (Int32)Math.Log2(numberOfWorkers * amountOfWork);
                        Utils.FillBalanced(tree, height);
                        tasks = Utils.GetDeletionTasks(tree, distribution);
                        break;
                    }
                default:
                    return;
            }

            Utils.RunAll(tasks);
        }
    }
}
