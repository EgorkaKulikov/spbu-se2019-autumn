using System;
using System.Linq;
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

        public static void Shake(Int32[] array) {
            var random = new Random();

            for (var i = 0; i < array.Length; i++) {
                var iswap = random.Next(0, array.Length - 1);
                var tmp = array[i];
                array[i] = array[iswap];
                array[iswap] = tmp;
            }
        }

        public static Int32[][] GetSimpleDistribution(Int32 numberOfWorkers, Int32 amountOfWork)
        {
            var array = Enumerable.Range(1, numberOfWorkers * amountOfWork).ToArray();
            Shake(array);

            var result = new Int32[numberOfWorkers][];

            for (Int32 i = 0; i < numberOfWorkers; i++)
            {
                result[i] = new Int32[amountOfWork];

                for (Int32 j = 0; j < amountOfWork; j++)
                {
                    result[i][j] = array[j + amountOfWork * i];
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
