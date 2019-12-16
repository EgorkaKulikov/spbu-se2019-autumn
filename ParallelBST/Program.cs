//.Freamwork 4.8

using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace ParallelTree
{
    class Program
    {
        static void Main(string[] args)
        {
            GoodParallelTree<int> goodTree = new GoodParallelTree<int>();
            //BadParallelTree<int>  badTree  = new BadParallelTree<int>();

            CrushTest(goodTree);
            //CrushTest(badTree);
        }

        private static void CrushTest(ICollection<int> col)
        {
            try
            {
                List<int> adding = new List<int>();

                for (int i = 0; i < 100; i++)
                    adding.Add(i);

                List<int> finding = new List<int>();

                for (int i = 0; i < 50; i++)
                    adding.Add(3 * i);

                List<int[]> list = new List<int[]>();

                for (int i = 0; i < 10; i++)
                    list.Add(new int[150]);

                List<Task> tasks = new List<Task>();

                foreach (var v in adding)
                    tasks.Add(Task.Run(() => col.Add(v)));

                foreach (var v in finding)
                    tasks.Add(Task.Run(() => col.Contains(v)));

                foreach (var a in list)
                    tasks.Add(Task.Run(() => col.CopyTo(a, 0)));

                for (int i = 0; i < 10; i++)
                    tasks.Add(Task.Run(() => col.Clear()));

                Task.WaitAll(tasks.ToArray());

                Console.WriteLine("Crush test passed!");
            }
            catch (Exception e)
            {
                Console.WriteLine("Whoops! Something wrong!");
                Console.WriteLine(e);
            }
        }
    }
}
