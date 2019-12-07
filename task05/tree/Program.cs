using System;
using System.Threading.Tasks;
using System.Linq;

namespace Task05
{
    class Program
    {
        static Int32 NextInt => Int32.Parse(Console.ReadLine());

        static void Main()
        {
            var tree = new CoarseTree<Int32, Int32>();
            var numberOfTasks = NextInt;
            var tasks = Enumerable.Range(0, numberOfTasks).Select(_ =>
            {
                var numberOfActions = NextInt;
                var actions = Enumerable.Range(0, numberOfActions).Select(_ =>
                {
                    var action = NextInt;
                    var index = NextInt;
                    var delay = NextInt;

                    return (action, index, delay);
                });

                return new Task(() =>
                {
                    foreach ((var action, var index, var delay) in actions)
                    {
                        Task.Delay(delay / 10);
                        switch (action)
                        {
                            case 0:
                                tree.Add(index, index + 1);
                                break;
                            case 1:
                                tree.Delete(index);
                                break;
                            case 2:
                                tree.Find(index);
                                break;
                        }
                    }
                });
            }).ToArray();

            foreach (var task in tasks)
            {
                task.Start();
            }

            Task.WaitAll(tasks);
        }
    }
}
