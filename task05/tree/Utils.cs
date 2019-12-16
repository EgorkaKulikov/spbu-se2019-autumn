using System;
using System.Threading.Tasks;

namespace Task05
{
    public static class Utils
    {
        public static void FillBalanced(ITree<Int32, Int32> tree, Int32 height, Int32 fillHeight)
        {
            Int32 size = (1 << height) - 1;
            Int32 maxDenumerator = 1 << fillHeight;

            for (Int32 denumerator = 2; denumerator <= maxDenumerator; denumerator <<= 1)
            {
                for (Int32 numerator = 1; numerator < denumerator; numerator += 2)
                {
                    var index = 1 + size * numerator / denumerator;
                    tree.Add(index, index + 1);
                }
            }
        }

        public static Int32[][] GetSimpleDistribution(Int32 numberOfWorkers, Int32 amountOfWork)
        {
            var result = new Int32[numberOfWorkers][];

            for (Int32 i = 0; i < numberOfWorkers; i++)
            {
                result[i] = new Int32[amountOfWork];

                for (Int32 j = 0; j < amountOfWork; j++)
                {
                    result[i][j] = 1 + j + amountOfWork * i;
                }
            }

            return result;
        }

        public static Int32[][] GetSimpleReverseDistribution(Int32 numberOfWorkers, Int32 amountOfWork)
        {
            var result = new Int32[numberOfWorkers][];

            for (Int32 i = 0; i < numberOfWorkers; i++)
            {
                result[i] = new Int32[amountOfWork];

                for (Int32 j = 0; j < amountOfWork; j++)
                {
                    result[i][j] = -(1 + j + amountOfWork * i);
                }
            }

            return result;
        }

        public static Task[] GetInsertionTasks(ITree<Int32, Int32> tree, Int32[][] distribution)
        {
            var tasks = new Task[distribution.Length];

            for (Int32 i = 0; i < distribution.Length; i++)
            {
                var indices = distribution[i];
                tasks[i] = new Task(() =>
                {
                    foreach (var index in indices)
                    {
                        tree.Add(index, index + 1);
                    }
                });
            }

            return tasks;
        }

        public static void RunAll(Task[] tasks)
        {
            foreach (var task in tasks)
            {
                task.Start();
            }

            Task.WaitAll(tasks);
        }
    }
}
